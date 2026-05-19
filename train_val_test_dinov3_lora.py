from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from my_model import SegmentationLoss
from my_model.metrics import AverageMeter, binary_metrics_from_logits, set_seed
from train_val_test_convnext_lora import (
    align_logits_and_masks,
    count_parameters,
    load_config,
    make_loader,
    merge_cli_config,
    resolve_config_path,
    resolve_path,
    save_checkpoint,
    save_visualization,
    str2bool,
)


warnings.filterwarnings("ignore")


DINOV3_MODEL_NAME = "dinov3_vitl16"
DINOV3_WEIGHT_NAME = "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
DINOV3_FEATURE_DIM = 1024
DINOV3_PATCH_SIZE = 16
DEFAULT_LORA_TARGETS = ("attn.qkv", "attn.proj")


def split_path_list(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item)]
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(value)] if value else []


def resolve_path_or_paths(value: Any, base_dir: Path) -> Any:
    paths = split_path_list(value)
    resolved = [resolve_path(path, base_dir) for path in paths]
    return resolved if isinstance(value, (list, tuple)) or (isinstance(value, str) and "," in value) else resolved[0]


def sample_paths_exist(value: Any) -> bool:
    paths = split_path_list(value)
    return bool(paths) and all(Path(path).exists() for path in paths)


def find_dinov3_repo(cfg: dict[str, Any], base_dir: Path) -> Path | None:
    candidates: list[Path] = []
    if cfg.get("dinov3_repo"):
        candidates.append(Path(str(cfg["dinov3_repo"])).expanduser())
    env_repo = os.environ.get("DINOV3_REPO")
    if env_repo:
        candidates.append(Path(str(env_repo)).expanduser())
    candidates.extend([base_dir / "dinov3", base_dir.parent / "dinov3", Path.cwd() / "dinov3"])

    for candidate in candidates:
        if not candidate.is_absolute():
            candidate = base_dir / candidate
        candidate = candidate.resolve()
        if (candidate / "hubconf.py").exists():
            return candidate
    return None


def load_dinov3_backbone(cfg: dict[str, Any], base_dir: Path) -> nn.Module:
    weights = Path(str(cfg["dinov3_weights"])).expanduser().resolve()
    repo = find_dinov3_repo(cfg, base_dir)
    if repo is not None:
        print(f"[dinov3] loading {DINOV3_MODEL_NAME} from local repo: {repo}")
        return torch.hub.load(str(repo), DINOV3_MODEL_NAME, source="local", weights=str(weights))

    if bool(cfg.get("allow_hub_download", False)):
        print("[dinov3] local repo not found, falling back to torch.hub github source")
        return torch.hub.load("facebookresearch/dinov3", DINOV3_MODEL_NAME, weights=str(weights))

    raise FileNotFoundError(
        "DINOv3 repo was not found. Clone facebookresearch/dinov3 and set dinov3_repo, "
        "or set allow_hub_download: true if this machine can access GitHub."
    )


def add_peft_lora_to_backbone(
    backbone: nn.Module,
    target_suffixes: tuple[str, ...],
    rank: int = 4,
    alpha: float = 8.0,
    dropout: float = 0.0,
) -> tuple[nn.Module, int]:
    try:
        from peft import LoraConfig, inject_adapter_in_model
    except ImportError as exc:
        raise ImportError(
            "use_lora=true requires the PEFT library. Install it in this environment with: "
            "pip install peft"
        ) from exc

    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=list(target_suffixes),
        lora_dropout=dropout,
        bias="none",
    )
    backbone = inject_adapter_in_model(config, backbone)
    lora_layers = sum(1 for _, module in backbone.named_modules() if hasattr(module, "lora_A"))
    return backbone, lora_layers


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class DINOv3HighFreqDecoder(nn.Module):
    def __init__(self, in_channels: int = DINOV3_FEATURE_DIM, decoder_channels: int = 256):
        super().__init__()
        if decoder_channels < 4:
            raise ValueError(f"decoder_channels must be >= 4, got {decoder_channels}")

        c1 = decoder_channels
        c2 = decoder_channels // 2
        c3 = decoder_channels // 4

        self.proj = ConvBNReLU(in_channels, c1)
        self.refine_8 = ConvBNReLU(c1, c2)
        self.high_freq = nn.Sequential(
            ConvBNReLU(3, c3),
            ConvBNReLU(c3, c3),
        )
        self.fuse_refine = nn.Sequential(
            ConvBNReLU(c2 + c3, c2),
            ConvBNReLU(c2, c3),
        )
        self.final_refine = ConvBNReLU(c3, c3)
        self.pred = nn.Conv2d(c3, 1, kernel_size=1)

    def forward(self, feats: torch.Tensor, frames: torch.Tensor) -> torch.Tensor:
        _, _, height, width = frames.shape
        x = self.proj(feats)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.refine_8(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

        high_freq = F.interpolate(frames, size=x.shape[-2:], mode="bilinear", align_corners=False)
        high_freq = self.high_freq(high_freq)
        x = torch.cat([x, high_freq], dim=1)
        x = self.fuse_refine(x)

        x = F.interpolate(x, size=(height, width), mode="bilinear", align_corners=False)
        x = self.final_refine(x)
        return self.pred(x)


class DINOv3ViTL16InpaintingDetector(nn.Module):
    """
    DINOv3 ViT-L/16 comparison model.

    This matches the official hub model named dinov3_vitl16 and the local
    weight file dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth.

    Input:
        clip: [B, T, 3, H, W], RGB in [0, 1]

    Output:
        logits: [B, T, 1, H, W]
    """

    def __init__(
        self,
        backbone: nn.Module,
        decoder_channels: int = 256,
        freeze_backbone: bool = True,
        use_lora: bool = False,
        lora_rank: int = 4,
        lora_alpha: float = 8.0,
        lora_dropout: float = 0.0,
        lora_targets: tuple[str, ...] = DEFAULT_LORA_TARGETS,
    ):
        super().__init__()
        self.backbone = backbone
        self.freeze_backbone = freeze_backbone
        self.use_lora = use_lora

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("image_mean", mean)
        self.register_buffer("image_std", std)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.lora_layers = 0
        if use_lora:
            if lora_rank <= 0:
                raise ValueError("lora_rank must be > 0 when use_lora is enabled")
            self.backbone, self.lora_layers = add_peft_lora_to_backbone(
                backbone=self.backbone,
                target_suffixes=lora_targets,
                rank=lora_rank,
                alpha=lora_alpha,
                dropout=lora_dropout,
            )
            if self.lora_layers == 0:
                raise RuntimeError(f"No DINOv3 Linear layers matched LoRA targets: {lora_targets}")

        self.decoder = DINOv3HighFreqDecoder(
            in_channels=DINOV3_FEATURE_DIM,
            decoder_channels=decoder_channels,
        )

    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        if clip.ndim != 5:
            raise ValueError(f"clip must have shape [B, T, 3, H, W], got {tuple(clip.shape)}")

        batch_size, num_frames, channels, height, width = clip.shape
        if channels != 3:
            raise ValueError(f"clip channel dimension must be 3, got {channels}")

        frames = clip.reshape(batch_size * num_frames, channels, height, width)
        normalized_frames = (frames - self.image_mean) / self.image_std
        if self.freeze_backbone and not self.use_lora:
            with torch.no_grad():
                feats = self.backbone.get_intermediate_layers(normalized_frames, n=1, reshape=True, norm=True)[0]
        else:
            feats = self.backbone.get_intermediate_layers(normalized_frames, n=1, reshape=True, norm=True)[0]
        logits = self.decoder(feats, frames)
        return logits.reshape(batch_size, num_frames, 1, height, width)


def validate_config(cfg: dict[str, Any], mode: str) -> None:
    errors = []
    if mode not in {"train", "val", "test"}:
        errors.append(f"type must be train/val/test, got {mode}")
    if int(cfg.get("input_size", 0)) <= 0:
        errors.append("input_size must be > 0")
    if int(cfg.get("input_size", 0)) % DINOV3_PATCH_SIZE != 0:
        errors.append(f"input_size must be divisible by {DINOV3_PATCH_SIZE} for DINOv3 ViT-L/16")
    if int(cfg.get("batch_size", 0)) <= 0:
        errors.append("batch_size must be > 0")
    if int(cfg.get("grad_accum_steps", 1)) <= 0:
        errors.append("grad_accum_steps must be > 0")
    if int(cfg.get("num_frames", 0)) <= 0 or int(cfg.get("num_frames", 0)) % 2 == 0:
        errors.append("num_frames must be a positive odd integer")

    required = ["val_samples"] if mode == "val" else ["test_samples"] if mode == "test" else ["train_samples", "val_samples"]
    for key in required:
        if not cfg.get(key) or not sample_paths_exist(cfg[key]):
            errors.append(f"{key} does not exist: {cfg.get(key)}")

    if not cfg.get("dinov3_weights") or not Path(cfg["dinov3_weights"]).exists():
        errors.append(f"dinov3_weights does not exist: {cfg.get('dinov3_weights')}")
    elif Path(cfg["dinov3_weights"]).name != DINOV3_WEIGHT_NAME:
        errors.append(f"dinov3_weights should be {DINOV3_WEIGHT_NAME} for {DINOV3_MODEL_NAME}")

    if cfg.get("dinov3_repo") and not (Path(cfg["dinov3_repo"]) / "hubconf.py").exists():
        errors.append(f"dinov3_repo does not look like facebookresearch/dinov3: {cfg['dinov3_repo']}")
    if cfg.get("checkpoint") and not Path(cfg["checkpoint"]).exists():
        errors.append(f"checkpoint does not exist: {cfg['checkpoint']}")
    if int(cfg.get("lora_rank", 0)) < 0:
        errors.append("lora_rank must be >= 0")

    if errors:
        raise ValueError("Config validation failed:\n  - " + "\n  - ".join(errors))


def build_model(cfg: dict[str, Any], device: torch.device, base_dir: Path) -> DINOv3ViTL16InpaintingDetector:
    backbone = load_dinov3_backbone(cfg, base_dir)
    lora_targets = tuple(str(item).strip() for item in str(cfg.get("lora_targets", "attn.qkv,attn.proj")).split(",") if str(item).strip())
    model = DINOv3ViTL16InpaintingDetector(
        backbone=backbone,
        decoder_channels=int(cfg.get("decoder_channels", 256)),
        freeze_backbone=bool(cfg.get("freeze_backbone", True)),
        use_lora=bool(cfg.get("use_lora", False)),
        lora_rank=int(cfg.get("lora_rank", 4)),
        lora_alpha=float(cfg.get("lora_alpha", 8.0)),
        lora_dropout=float(cfg.get("lora_dropout", 0.0)),
        lora_targets=lora_targets,
    )

    checkpoint = cfg.get("checkpoint", "")
    if checkpoint:
        state = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)
        print(f"[checkpoint] loaded: {checkpoint}")

    total, trainable = count_parameters(model)
    print(f"[model] DINOv3 ViT-L/16 pretrained backbone, LoRA layers: {model.lora_layers}")
    print(f"[params] total {total:,} trainable {trainable:,}")
    return model.to(device)


LOG_FIELDS = [
    "epoch",
    "lr",
    "train_loss",
    "train_bce_loss",
    "train_focal_loss",
    "train_iou_loss",
    "train_iou",
    "train_f1",
    "train_precision",
    "train_recall",
    "train_accuracy",
    "val_loss",
    "val_bce_loss",
    "val_focal_loss",
    "val_iou_loss",
    "val_iou",
    "val_f1",
    "val_precision",
    "val_recall",
    "val_accuracy",
]


def summarize_samples(value: Any) -> str:
    paths = split_path_list(value)
    return ";".join(Path(path).name for path in paths)


def init_training_log(path: Path, cfg: dict[str, Any], model: nn.Module) -> None:
    total, trainable = count_parameters(model)
    lines = [
        f"# model={DINOV3_MODEL_NAME} use_lora={bool(cfg.get('use_lora', False))} "
        f"lora_rank={int(cfg.get('lora_rank', 4))} input_size={int(cfg.get('input_size', 0))} "
        f"num_frames={int(cfg.get('num_frames', 0))}",
        f"# train_samples={summarize_samples(cfg.get('train_samples', ''))} "
        f"val_samples={summarize_samples(cfg.get('val_samples', ''))}",
        f"# batch_size={int(cfg.get('batch_size', 0))} lr={float(cfg.get('learning_rate', 0.0))} "
        f"params_total={total} params_trainable={trainable}",
        ",".join(LOG_FIELDS),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_log_value(value: float | int | None) -> str:
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.8g}"


def append_epoch_log(
    path: Path,
    epoch: int,
    lr: float,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float] | None = None,
) -> None:
    row_values: dict[str, float | int | None] = {"epoch": epoch, "lr": lr}
    for key in ["loss", "bce_loss", "focal_loss", "iou_loss", "iou", "f1", "precision", "recall", "accuracy"]:
        row_values[f"train_{key}"] = train_metrics.get(key)
        row_values[f"val_{key}"] = val_metrics.get(key) if val_metrics is not None else None

    with open(path, "a", encoding="utf-8") as f:
        f.write(",".join(format_log_value(row_values.get(field)) for field in LOG_FIELDS) + "\n")


def forward_in_frame_chunks(model: nn.Module, frames: torch.Tensor, frame_chunk: int) -> torch.Tensor:
    if frame_chunk <= 0 or frames.shape[1] <= frame_chunk:
        return model(frames)

    outputs = []
    for start in range(0, frames.shape[1], frame_chunk):
        outputs.append(model(frames[:, start : start + frame_chunk]))
    return torch.cat(outputs, dim=1)


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
    grad_accum_steps: int = 1,
    eval_frame_chunk: int = 0,
) -> dict[str, float]:
    is_train = mode == "train"
    model.train(is_train)

    meters = {
        key: AverageMeter()
        for key in ["loss", "bce_loss", "focal_loss", "iou_loss", "iou", "f1", "precision", "recall", "accuracy"]
    }

    if is_train:
        if optimizer is None:
            raise ValueError("optimizer is required in train mode")
        optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader, start=1):
        frames, masks = batch[0].to(device), batch[1].to(device)
        batch_size = frames.shape[0]

        with torch.set_grad_enabled(is_train):
            with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
                logits_all = model(frames) if is_train else forward_in_frame_chunks(model, frames, eval_frame_chunk)
                logits, loss_masks = align_logits_and_masks(logits_all, masks)
                loss, loss_items = criterion(logits, loss_masks)

            if is_train:
                scaled_loss = loss / grad_accum_steps
                should_step = step % grad_accum_steps == 0 or step == len(loader)
                if scaler is not None and amp and device.type == "cuda":
                    scaler.scale(scaled_loss).backward()
                    if should_step:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                else:
                    scaled_loss.backward()
                    if should_step:
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)

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


def train(cfg: dict[str, Any], device: torch.device, base_dir: Path) -> None:
    save_dir = Path(cfg["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    model = build_model(cfg, device, base_dir)
    log_path = save_dir / "log.txt"
    init_training_log(log_path, cfg, model)
    print(f"[log] writing training metrics to {log_path}")

    criterion = SegmentationLoss()
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters. Enable decoder training or LoRA.")

    optimizer = Adam(
        trainable_params,
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
        betas=(0.9, 0.999),
    )
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
            grad_accum_steps=int(cfg.get("grad_accum_steps", 1)),
        )
        print(
            f"[epoch {epoch}] train loss {train_metrics['loss']:.4f} "
            f"f1 {train_metrics['f1']:.4f} iou {train_metrics['iou']:.4f}"
        )

        val_metrics = None
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
                    eval_frame_chunk=int(cfg.get("eval_frame_chunk", 1)),
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

        append_epoch_log(log_path, epoch, optimizer.param_groups[0]["lr"], train_metrics, val_metrics)


def evaluate(cfg: dict[str, Any], device: torch.device, mode: str, base_dir: Path) -> None:
    model = build_model(cfg, device, base_dir)
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
            eval_frame_chunk=int(cfg.get("eval_frame_chunk", 1)),
        )
    print(
        f"[{mode}] loss {metrics['loss']:.4f} "
        f"f1 {metrics['f1']:.4f} iou {metrics['iou']:.4f} "
        f"precision {metrics['precision']:.4f} recall {metrics['recall']:.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="DINOv3 ViT-L/16 pretrained comparison train/val/test")
    parser.add_argument("--config", type=str, default="configs/dinov3_vitl16_lora.yml")
    parser.add_argument("--type", type=str, default=None, choices=["train", "val", "test"])
    parser.add_argument("--train_samples", type=str, default=None)
    parser.add_argument("--val_samples", type=str, default=None)
    parser.add_argument("--test_samples", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--dinov3_weights", type=str, default=None)
    parser.add_argument("--dinov3_repo", type=str, default=None)
    parser.add_argument("--allow_hub_download", type=str2bool, default=None)
    parser.add_argument("--input_size", type=int, default=None)
    parser.add_argument("--num_frames", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--grad_accum_steps", type=int, default=None)
    parser.add_argument("--eval_frame_chunk", type=int, default=None)
    parser.add_argument("--n_epochs", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--decoder_channels", type=int, default=None)
    parser.add_argument("--freeze_backbone", type=str2bool, default=None)
    parser.add_argument("--use_lora", type=str2bool, default=None)
    parser.add_argument("--lora_rank", type=int, default=None)
    parser.add_argument("--lora_alpha", type=float, default=None)
    parser.add_argument("--lora_dropout", type=float, default=None)
    parser.add_argument("--lora_targets", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--visualization_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    cfg_path = resolve_config_path(args.config)
    cfg = merge_cli_config(load_config(str(cfg_path)), args)
    base_dir = cfg_path.parent.parent

    defaults = {
        "dinov3_weights": DINOV3_WEIGHT_NAME,
        "dinov3_repo": "",
        "allow_hub_download": False,
        "freeze_backbone": True,
        "use_lora": False,
        "lora_rank": 4,
        "lora_alpha": 8.0,
        "lora_dropout": 0.0,
        "lora_targets": ",".join(DEFAULT_LORA_TARGETS),
        "decoder_channels": 256,
        "save_dir": "runs/dinov3_vitl16",
        "visualization_dir": "runs/dinov3_vitl16/vis",
        "checkpoint": "",
        "grad_accum_steps": 1,
        "eval_frame_chunk": 1,
    }
    for key, value in defaults.items():
        cfg.setdefault(key, value)

    for key in [
        "train_samples",
        "val_samples",
        "test_samples",
        "dinov3_weights",
        "dinov3_repo",
        "checkpoint",
        "save_dir",
        "visualization_dir",
    ]:
        if key in cfg and cfg[key]:
            cfg[key] = resolve_path_or_paths(cfg[key], base_dir) if key.endswith("_samples") else resolve_path(str(cfg[key]), base_dir)

    mode = cfg.get("type", "train")
    validate_config(cfg, mode)

    set_seed(int(cfg.get("seed", 666666)))
    device = torch.device(cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[device] {device}")
    print(f"[mode] {mode}")
    print(f"[dinov3] model={DINOV3_MODEL_NAME} weights={cfg['dinov3_weights']}")
    print(f"[lora] enabled={bool(cfg.get('use_lora', False))} rank={int(cfg.get('lora_rank', 4))}")

    if mode == "train":
        train(cfg, device, base_dir)
    else:
        evaluate(cfg, device, mode, base_dir)


if __name__ == "__main__":
    main()
