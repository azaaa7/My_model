from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

from my_model import SegmentationLoss
from my_model.metrics import AverageMeter, binary_metrics_from_logits, set_seed
from zzz_dataset_toolkit import build_dataloader


warnings.filterwarnings("ignore")


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid bool value: {value}")


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


def resolve_config_path(path: str) -> Path:
    cfg_path = Path(path).expanduser()
    if cfg_path.is_absolute() or cfg_path.exists():
        return cfg_path.resolve()

    script_relative = Path(__file__).resolve().parent / cfg_path
    if script_relative.exists():
        return script_relative
    return cfg_path.resolve()


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

    if cfg.get("checkpoint") and not Path(cfg["checkpoint"]).exists():
        errors.append(f"checkpoint does not exist: {cfg['checkpoint']}")
    if int(cfg.get("lora_rank", 0)) < 0:
        errors.append("lora_rank must be >= 0")

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
        val_num_frames=cfg.get("val_num_frames", 0),
        dataset_repeat=cfg.get("dataset_repeat", 1),
        robust_noise_snr=cfg.get("robust_noise_snr", 0),
        robust_jpeg_quality=cfg.get("robust_jpeg_quality", 0),
        num_clips=cfg.get("num_clips", 1),
        clip_stride=cfg.get("clip_stride", 1),
    )


class LoRALinear(nn.Module):
    def __init__(
        self,
        base: nn.Linear,
        rank: int = 4,
        alpha: float = 8.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")

        self.base = base
        self.rank = rank
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=np.sqrt(5))

        for param in self.base.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lora = F.linear(F.linear(self.dropout(x), self.lora_a), self.lora_b) * self.scaling
        return self.base(x) + lora


def add_lora_to_linears(
    module: nn.Module,
    rank: int = 4,
    alpha: float = 8.0,
    dropout: float = 0.0,
    name_prefix: str = "",
) -> int:
    count = 0
    for name, child in list(module.named_children()):
        full_name = f"{name_prefix}.{name}" if name_prefix else name
        if isinstance(child, nn.Linear):
            setattr(module, name, LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout))
            count += 1
        else:
            count += add_lora_to_linears(child, rank=rank, alpha=alpha, dropout=dropout, name_prefix=full_name)
    return count


class ConvNeXtTinyInpaintingDetector(nn.Module):
    """
    ConvNeXt-Tiny comparison model.

    Input:
        clip: [B, T, 3, H, W], RGB in [0, 1]

    Output:
        logits: [B, T, 1, H, W]
    """

    def __init__(
        self,
        decoder_channels: int = 256,
        freeze_backbone: bool = True,
        use_lora: bool = False,
        lora_rank: int = 4,
        lora_alpha: float = 8.0,
        lora_dropout: float = 0.0,
    ):
        super().__init__()
        weights = ConvNeXt_Tiny_Weights.DEFAULT
        model = convnext_tiny(weights=weights)
        self.backbone = model.features

        mean = torch.tensor(weights.transforms().mean).view(1, 3, 1, 1)
        std = torch.tensor(weights.transforms().std).view(1, 3, 1, 1)
        self.register_buffer("image_mean", mean)
        self.register_buffer("image_std", std)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.lora_layers = 0
        if use_lora:
            if lora_rank <= 0:
                raise ValueError("lora_rank must be > 0 when use_lora is enabled")
            self.lora_layers = add_lora_to_linears(
                self.backbone,
                rank=lora_rank,
                alpha=lora_alpha,
                dropout=lora_dropout,
            )

        self.decoder = nn.Sequential(
            nn.Conv2d(768, decoder_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_channels, decoder_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_channels // 2, 1, kernel_size=1),
        )

    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        if clip.ndim != 5:
            raise ValueError(f"clip must have shape [B, T, 3, H, W], got {tuple(clip.shape)}")

        batch_size, num_frames, channels, height, width = clip.shape
        if channels != 3:
            raise ValueError(f"clip channel dimension must be 3, got {channels}")

        frames = clip.reshape(batch_size * num_frames, channels, height, width)
        frames = (frames - self.image_mean) / self.image_std
        feats = self.backbone(frames)
        logits = self.decoder(feats)
        logits = F.interpolate(logits, size=(height, width), mode="bilinear", align_corners=False)
        return logits.reshape(batch_size, num_frames, 1, height, width)


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return total, trainable


def build_model(cfg: dict[str, Any], device: torch.device) -> ConvNeXtTinyInpaintingDetector:
    model = ConvNeXtTinyInpaintingDetector(
        decoder_channels=int(cfg.get("decoder_channels", 256)),
        freeze_backbone=bool(cfg.get("freeze_backbone", True)),
        use_lora=bool(cfg.get("use_lora", False)),
        lora_rank=int(cfg.get("lora_rank", 4)),
        lora_alpha=float(cfg.get("lora_alpha", 8.0)),
        lora_dropout=float(cfg.get("lora_dropout", 0.0)),
    )
    checkpoint = cfg.get("checkpoint", "")
    if checkpoint:
        state = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)
        print(f"[checkpoint] loaded: {checkpoint}")

    total, trainable = count_parameters(model)
    print(f"[model] ConvNeXt-Tiny pretrained backbone, LoRA layers: {model.lora_layers}")
    print(f"[params] total {total:,} trainable {trainable:,}")
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


def save_checkpoint(path: Path, model: nn.Module, optimizer: Adam, epoch: int, metrics: dict[str, float]) -> None:
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
        cv2.imwrite(str(out_dir / f"{stem}_vis.jpg"), canvas)


def run_epoch(
    model: nn.Module,
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

     
    meters = {key: AverageMeter() for key in ["loss", "bce_loss", "focal_loss", "iou_loss", "iou", "f1", "precision", "recall", "accuracy"]}

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
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = Adam(trainable_params, lr=float(cfg["learning_rate"]), weight_decay=float(cfg.get("weight_decay", 0.0)), betas=(0.9, 0.999))
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
    parser = argparse.ArgumentParser(description="ConvNeXt-Tiny pretrained comparison train/val/test")
    parser.add_argument("--config", type=str, default="configs/convnext_tiny_lora.yml")
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
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--decoder_channels", type=int, default=None)
    parser.add_argument("--freeze_backbone", type=str2bool, default=None)
    parser.add_argument("--use_lora", type=str2bool, default=None)
    parser.add_argument("--lora_rank", type=int, default=None)
    parser.add_argument("--lora_alpha", type=float, default=None)
    parser.add_argument("--lora_dropout", type=float, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--visualization_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    cfg_path = resolve_config_path(args.config)
    cfg = merge_cli_config(load_config(str(cfg_path)), args)
    base_dir = cfg_path.parent.parent

    defaults = {
        "freeze_backbone": True,
        "use_lora": False,
        "lora_rank": 4,
        "lora_alpha": 8.0,
        "lora_dropout": 0.0,
        "decoder_channels": 256,
        "save_dir": "runs/convnext_tiny",
        "visualization_dir": "runs/convnext_tiny/vis",
        "checkpoint": "",
    }
    for key, value in defaults.items():
        cfg.setdefault(key, value)

    for key in ["train_samples", "val_samples", "test_samples", "checkpoint", "save_dir", "visualization_dir"]:
        if key in cfg and cfg[key]:
            cfg[key] = resolve_path(str(cfg[key]), base_dir)

    mode = cfg.get("type", "train")
    validate_config(cfg, mode)

    set_seed(int(cfg.get("seed", 666666)))
    device = torch.device(cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[device] {device}")
    print(f"[mode] {mode}")
    print(f"[lora] enabled={bool(cfg.get('use_lora', False))} rank={int(cfg.get('lora_rank', 4))}")

    if mode == "train":
        train(cfg, device)
    else:
        evaluate(cfg, device, mode)


if __name__ == "__main__":
    main()
