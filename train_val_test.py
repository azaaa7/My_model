from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")
import argparse
import json
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from my_model import SegmentationLoss, SimpleHRNetInpaintingDetector
from my_model.metrics import AverageMeter, binary_metrics_from_logits, set_seed
from zzz_dataset_toolkit import build_dataloader


def load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def merge_cli_config(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    for key, value in vars(args).items():
        if key != "config" and value is not None:
            cfg[key] = value
    return cfg


def resolve_path(path: str, base_dir: Path) -> str:
    if not path:
        return path
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = base_dir / p
    return str(p.resolve())


def validate_config(cfg: dict[str, Any], mode: str) -> None:
    errors = []
    if mode not in {"train", "val", "test"}:
        errors.append(f"type must be train/val/test, got {mode}")
    if int(cfg.get("input_size", 0)) <= 0:
        errors.append("input_size must be > 0")
    if int(cfg.get("batch_size", 0)) <= 0:
        errors.append("batch_size must be > 0")
    if int(cfg.get("num_frames", 0)) <= 0 or int(cfg.get("num_frames", 0)) % 2 == 0:
        errors.append("num_frames must be a positive odd integer")

    required = ["val_samples"] if mode == "val" else ["test_samples"] if mode == "test" else ["train_samples", "val_samples"]
    for key in required:
        if not cfg.get(key) or not Path(cfg[key]).exists():
            errors.append(f"{key} does not exist: {cfg.get(key)}")

    if cfg.get("hrnet_path") and not Path(cfg["hrnet_path"]).exists():
        errors.append(f"hrnet_path does not exist: {cfg['hrnet_path']}")
    if cfg.get("checkpoint") and not Path(cfg["checkpoint"]).exists():
        errors.append(f"checkpoint does not exist: {cfg['checkpoint']}")

    if errors:
        raise ValueError("Config validation failed:\n  - " + "\n  - ".join(errors))


def make_loader(cfg: dict[str, Any], mode: str):
    samples_key = "test_samples" if mode == "test" else "val_samples" if mode == "val" else "train_samples"
    return build_dataloader(
        samples=cfg[samples_key],
        mode=mode,
        batch_size=cfg["batch_size"] if mode == "train" else 1,
        num_workers=cfg["num_workers"],
        input_size=cfg["input_size"],
        gt_ratio=cfg["gt_ratio"],
        num_frames=cfg["num_frames"],
        robust_noise_snr=cfg.get("robust_noise_snr", 0),
        robust_jpeg_quality=cfg.get("robust_jpeg_quality", 0),
    )


def build_model(cfg: dict[str, Any], device: torch.device) -> SimpleHRNetInpaintingDetector:
    model = SimpleHRNetInpaintingDetector(
        hrnet_path=cfg.get("hrnet_path") or None,
        hrnet_extra_name=cfg.get("hrnet_extra_name", "w32_extra"),
        freeze_backbone=bool(cfg.get("freeze_backbone", False)),
        decoder_channels=int(cfg.get("decoder_channels", 32)),
    )
    checkpoint = cfg.get("checkpoint", "")
    if checkpoint:
        state = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)
        print(f"[checkpoint] loaded: {checkpoint}")
    return model.to(device)


def center_logits(logits: torch.Tensor) -> torch.Tensor:
    return logits[:, logits.shape[1] // 2]


def align_logits_and_masks(logits_all: torch.Tensor, masks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if masks.ndim == 5:
        b, t, c, h, w = masks.shape
        logits = logits_all.reshape(b * t, 1, logits_all.shape[-2], logits_all.shape[-1])
        masks = masks.reshape(b * t, c, h, w)
    else:
        logits = center_logits(logits_all)

    if logits.shape[-2:] != masks.shape[-2:]:
        logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
    return logits, masks


def save_checkpoint(path: Path, model: torch.nn.Module, optimizer: Adam, epoch: int, metrics: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "metrics": metrics,
        },
        path,
    )


def save_visualization(
    frames: torch.Tensor,
    logits: torch.Tensor,
    target: torch.Tensor,
    names: list[str] | tuple[str, ...],
    out_dir: Path,
    threshold: float,
    max_items: int = 50,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if target.ndim == 4 and logits.ndim == 5:
        logits = logits[:, logits.shape[1] // 2]
    probs = torch.sigmoid(logits).detach().cpu()
    pred = (probs > threshold).float()
    frames = frames.detach().cpu()
    target = target.detach().cpu()

    for idx in range(min(frames.shape[0], max_items)):
        if target.ndim == 5:
            frame_idx = target.shape[1] // 2
            image = frames[idx, frame_idx].permute(1, 2, 0).numpy()
            gt = target[idx, frame_idx, 0].numpy()
            pd = pred[idx, frame_idx, 0].numpy()
            prob = probs[idx, frame_idx, 0].numpy()
        else:
            image = frames[idx, frames.shape[1] // 2].permute(1, 2, 0).numpy()
            gt = target[idx, 0].numpy()
            pd = pred[idx, 0].numpy()
            prob = probs[idx, 0].numpy()

        image = np.ascontiguousarray(np.clip(image * 255.0, 0, 255).astype(np.uint8))
        gt = np.ascontiguousarray((gt * 255.0).astype(np.uint8))
        pd = np.ascontiguousarray((pd * 255.0).astype(np.uint8))
        prob = np.ascontiguousarray(np.clip(prob * 255.0, 0, 255).astype(np.uint8))
        if gt.shape[:2] != image.shape[:2]:
            size = image.shape[1], image.shape[0]
            gt = cv2.resize(gt, size, interpolation=cv2.INTER_NEAREST)
            pd = cv2.resize(pd, size, interpolation=cv2.INTER_NEAREST)
            prob = cv2.resize(prob, size, interpolation=cv2.INTER_LINEAR)

        gt_rgb = cv2.cvtColor(gt, cv2.COLOR_GRAY2RGB)
        pd_rgb = cv2.cvtColor(pd, cv2.COLOR_GRAY2RGB)
        prob_rgb = cv2.cvtColor(prob, cv2.COLOR_GRAY2RGB)
        canvas = np.concatenate([image[..., ::-1], prob_rgb, pd_rgb, gt_rgb], axis=1)
        stem = Path(str(names[idx])).stem if idx < len(names) else f"sample_{idx:04d}"
        name = f"{stem}_vis.jpg"
        cv2.imwrite(str(out_dir / name), canvas)


def run_epoch(
    model: torch.nn.Module,
    loader,
    criterion: SegmentationLoss,
    device: torch.device,
    mode: str,
    optimizer: Adam | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    amp: bool = False,
    threshold: float = 0.5,
    visualization_dir: Path | None = None,
) -> dict[str, float]:
    is_train = mode == "train"
    model.train(is_train)

    meters = {key: AverageMeter() for key in ["loss", "focal_loss", "bce_loss", "iou_loss", "iou", "f1", "precision", "recall", "accuracy"]}

    for step, batch in enumerate(loader, start=1):
        frames, masks = batch[0].to(device), batch[1].to(device)
        batch_size = frames.shape[0]

        if is_train:
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
                logits_all = model(frames)
                logits, loss_masks = align_logits_and_masks(logits_all, masks)
                loss, loss_items = criterion(logits, loss_masks)

            if is_train:
                if scaler is not None and amp and device.type == "cuda":
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        metric_items = binary_metrics_from_logits(logits.detach(), loss_masks.detach(), threshold=threshold)
        for key, value in {**loss_items, **metric_items}.items():
            meters[key].update(value, batch_size)

        if visualization_dir is not None and not is_train and step <= 50:
            names = batch[4] if len(batch) > 4 else []
            save_visualization(frames, logits_all.detach(), masks, names, visualization_dir, threshold, max_items=frames.shape[0])

        if is_train and (step == 1 or step % 20 == 0):
            print(
                f"[train] step {step:04d}/{len(loader):04d} "
                f"loss {meters['loss'].avg:.4f} iou {meters['iou'].avg:.4f} f1 {meters['f1'].avg:.4f}"
            )

    return {key: meter.avg for key, meter in meters.items()}


def train(cfg: dict[str, Any], device: torch.device) -> None:
    save_dir = Path(cfg["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    model = build_model(cfg, device)
    criterion = SegmentationLoss()
    optimizer = Adam(model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg.get("weight_decay", 0.0)), betas=(0.9, 0.999))
    scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5, mode="max")
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg.get("amp", True)) and device.type == "cuda")

    train_loader = make_loader(cfg, "train")
    val_loader = make_loader(cfg, "val")

    best_iou = -1.0
    for epoch in range(1, int(cfg["n_epochs"]) + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            "train",
            optimizer=optimizer,
            scaler=scaler,
            amp=bool(cfg.get("amp", True)),
            threshold=float(cfg.get("threshold", 0.5)),
        )
        print(
            f"[epoch {epoch}] train loss {train_metrics['loss']:.4f} "
            f"f1 {train_metrics['f1']:.4f} iou {train_metrics['iou']:.4f}"
        )

        if epoch % int(cfg.get("validate_every", 1)) == 0:
            with torch.no_grad():
                val_metrics = run_epoch(
                    model,
                    val_loader,
                    criterion,
                    device,
                    "val",
                    amp=False,
                    threshold=float(cfg.get("threshold", 0.5)),
                )
            scheduler.step(val_metrics["iou"])
            print(
                f"[epoch {epoch}] val loss {val_metrics['loss']:.4f} "
                f"f1 {val_metrics['f1']:.4f} iou {val_metrics['iou']:.4f}"
            )

            save_checkpoint(save_dir / "latest.pt", model, optimizer, epoch, val_metrics)
            if val_metrics["iou"] > best_iou:
                best_iou = val_metrics["iou"]
                save_checkpoint(save_dir / "best_iou.pt", model, optimizer, epoch, val_metrics)
                print(f"[checkpoint] best_iou updated: {best_iou:.4f}")


def evaluate(cfg: dict[str, Any], device: torch.device, mode: str) -> None:
    model = build_model(cfg, device)
    criterion = SegmentationLoss()
    loader = make_loader(cfg, mode)
    vis_dir = Path(cfg["visualization_dir"]) / mode if cfg.get("visualization_dir") else None

    with torch.no_grad():
        metrics = run_epoch(
            model,
            loader,
            criterion,
            device,
            mode,
            amp=False,
            threshold=float(cfg.get("threshold", 0.5)),
            visualization_dir=vis_dir,
        )
    print(
        f"[{mode}] loss {metrics['loss']:.4f} "
        f"f1 {metrics['f1']:.4f} iou {metrics['iou']:.4f} "
        f"precision {metrics['precision']:.4f} recall {metrics['recall']:.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple HRNet inpainting detector train/val/test")
    parser.add_argument("--config", type=str, default="configs/default.yml")
    parser.add_argument("--type", type=str, default=None, choices=["train", "val", "test"])
    parser.add_argument("--train_samples", type=str, default=None)
    parser.add_argument("--val_samples", type=str, default=None)
    parser.add_argument("--test_samples", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--input_size", type=int, default=None)
    parser.add_argument("--num_frames", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--n_epochs", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--visualization_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg = merge_cli_config(load_config(str(cfg_path)), args)
    base_dir = cfg_path.parent.parent

    for key in ["train_samples", "val_samples", "test_samples", "hrnet_path", "checkpoint", "save_dir", "visualization_dir"]:
        if key in cfg and cfg[key]:
            cfg[key] = resolve_path(str(cfg[key]), base_dir)

    mode = cfg.get("type", "train")
    validate_config(cfg, mode)

    set_seed(int(cfg.get("seed", 666666)))
    device = torch.device(cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[device] {device}")
    print(f"[mode] {mode}")

    if mode == "train":
        train(cfg, device)
    else:
        evaluate(cfg, device, mode)


if __name__ == "__main__":
    main()
