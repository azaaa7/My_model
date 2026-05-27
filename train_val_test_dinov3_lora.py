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
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, ReduceLROnPlateau, SequentialLR

from my_model import SegmentationLoss
from my_model.dinov3_dpt_fpn import DPTReassembleNeck, FPNDecoder
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
    lora_layers: list[int] | None = None,
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
    if lora_layers is not None and len(lora_layers) > 0:
        # PEFT's layers_to_transform uses hf-style naming (model.layers.X)
        # which is incompatible with DINOv3's blocks.X naming.  Generate
        # explicit per-block target names for suffix matching instead.
        explicit_targets = []
        for idx in lora_layers:
            for suffix in target_suffixes:
                explicit_targets.append(f"blocks.{idx}.{suffix}")
        config.target_modules = explicit_targets
    backbone = inject_adapter_in_model(config, backbone)
    lora_count = sum(1 for _, module in backbone.named_modules() if hasattr(module, "lora_A"))
    return backbone, lora_count


def parse_lora_layers(value: str) -> list[int] | None:
    """Parse lora_layers config string to list of ints or None.

    "all" / ""  →  None (all layers)
    "5-23"      →  [5, 6, ..., 23]
    "5,11,17,23"→  [5, 11, 17, 23]
    """
    if not value or str(value).strip().lower() in ("all", ""):
        return None
    value = str(value).strip()
    if "-" in value and "," not in value:
        parts = value.split("-")
        if len(parts) == 2:
            return list(range(int(parts[0]), int(parts[1]) + 1))
    return [int(x.strip()) for x in value.split(",") if x.strip().isdigit()]


# ── DINOv3 Single-layer Patch Encoder ─────────────────────────────────────

class DinoSingleLayerEncoder(nn.Module):
    """Extract the last layer DINOv3 patch tokens without channel modification.

    Only extracts the final ViT block's patch token features at 1024-dim,
    no multi-layer fusion or channel projection.

    Args:
        backbone: DINOv3 VisionTransformer
        last_layer_idx: 0-indexed block index to extract (default 23 for ViT-L/24)
        freeze_backbone: whether backbone params require grad
        use_lora: whether LoRA was injected on backbone
    """
    def __init__(
        self,
        backbone: nn.Module,
        last_layer_idx: int = 23,
        freeze_backbone: bool = True,
        use_lora: bool = False,
    ):
        super().__init__()
        self.backbone = backbone
        self.last_layer_idx = last_layer_idx
        self.freeze_backbone = freeze_backbone
        self.use_lora = use_lora

        # ImageNet normalization buffers
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("image_mean", mean)
        self.register_buffer("image_std", std)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """Extract last-layer patch features.

        Args:
            video: [B, T, 3, H, W] RGB in [0, 1]

        Returns:
            features: [B, T, 1024, H//16, W//16]
        """
        B, T, C, H, W = video.shape
        frames = video.reshape(B * T, C, H, W)
        normalized = (frames - self.image_mean) / self.image_std

        with torch.set_grad_enabled(not (self.freeze_backbone and not self.use_lora)):
            layer_outputs = self.backbone.get_intermediate_layers(
                normalized, n=[self.last_layer_idx], reshape=True, norm=True
            )

        features = layer_outputs[0]  # [B*T, 1024, H/16, W/16]
        return features.reshape(B, T, *features.shape[1:])


# ── Simple Segmentation Head (DINOv3-IML style) ───────────────────────────

class SimpleSegHead(nn.Module):
    """Lightweight conv head from DINOv3-IML (Irennnne et al., 2026).

    Architecture:
        Conv3×3 (feat_dim → feat_dim/2) → BN → ReLU
        Conv3×3 (feat_dim/2 → feat_dim/4) → BN → ReLU
        Conv1×1 (feat_dim/4 → 1)
        → bilinear upsample to input_size

    Args:
        feat_dim: DINOv3 feature dimension (1024 for ViT-L)
        input_size: target mask resolution (default 512)
    """

    def __init__(self, feat_dim: int = 1024, input_size: int = 512):
        super().__init__()
        self.input_size = input_size
        self.head = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(feat_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_dim // 2, feat_dim // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(feat_dim // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_dim // 4, 1, kernel_size=1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, encoder_feature_t: torch.Tensor) -> torch.Tensor:
        """Decode mask from encoder feature.

        Args:
            encoder_feature_t: [B, 1024, h, w] single-frame encoder output

        Returns:
            mask_logits: [B, 1, input_size, input_size]
        """
        x = self.head(encoder_feature_t)  # [B, 1, h, w]
        x = F.interpolate(x, size=(self.input_size, self.input_size),
                          mode="bilinear", align_corners=False)
        return x


# ── DINOv3 Multi-layer Patch Encoder ──────────────────────────────────────

class DinoMultiLayerEncoder(nn.Module):
    """Extract multiple DINOv3 block patch token outputs for DPT neck.

    Extracts token maps from specified ViT blocks (all at 32×32, 1024-dim),
    suitable as input to DPTReassembleNeck.

    Args:
        backbone: DINOv3 VisionTransformer (with LoRA already injected)
        extract_layers: 0-indexed block indices, e.g. (5, 11, 17, 23)
        freeze_backbone: whether backbone params require grad
        use_lora: whether LoRA was injected on backbone
    """

    def __init__(
        self,
        backbone: nn.Module,
        extract_layers: tuple[int, ...] = (5, 11, 17, 23),
        freeze_backbone: bool = True,
        use_lora: bool = False,
    ):
        super().__init__()
        self.backbone = backbone
        self.extract_layers = list(extract_layers)
        self.freeze_backbone = freeze_backbone
        self.use_lora = use_lora

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("image_mean", mean)
        self.register_buffer("image_std", std)

    def forward(self, video: torch.Tensor) -> dict[int, torch.Tensor]:
        """Extract multi-layer patch features.

        Args:
            video: [B, T, 3, H, W] RGB in [0, 1]

        Returns:
            features: {layer_idx: [B, T, 1024, H//16, W//16], ...}
        """
        B, T, C, H, W = video.shape
        frames = video.reshape(B * T, C, H, W)
        normalized = (frames - self.image_mean) / self.image_std

        with torch.set_grad_enabled(not (self.freeze_backbone and not self.use_lora)):
            layer_outputs = self.backbone.get_intermediate_layers(
                normalized, n=self.extract_layers, reshape=True, norm=True,
            )

        # layer_outputs is tuple of [B*T, 1024, H/16, W/16] per layer
        feats: dict[int, torch.Tensor] = {}
        for idx, feat in zip(self.extract_layers, layer_outputs):
            feats[idx] = feat.reshape(B, T, *feat.shape[1:])

        return feats


class DINOv3ViTL16InpaintingDetector(nn.Module):
    """
    DINOv3 ViT-L/16 model with single-layer encoder and progressive upsampling decoder.

    Extracts the last ViT block's patch tokens at 1024-dim (no channel projection),
    then decodes via progressive 2× upsampling with ConvBlock at each scale:
        1024(1/16) → 512(1/8) → 256(1/4) → 128(1/2) → 64(1/2) → 32(1/1) → 1 (mask)

    Input:
        clip: [B, T, 3, H, W], RGB in [0, 1]

    Output:
        mask_logits: [B, T, 1, H, W]
    """

    def __init__(
        self,
        backbone: nn.Module,
        last_layer_idx: int = 23,
        freeze_backbone: bool = True,
        use_lora: bool = False,
        lora_rank: int = 4,
        lora_alpha: float = 8.0,
        lora_dropout: float = 0.0,
        lora_targets: tuple[str, ...] = DEFAULT_LORA_TARGETS,
        *,
        use_dpt_fpn: bool = False,
        extract_layers: tuple[int, ...] = (5, 11, 17, 23),
        neck_channels: int = 256,
        lora_block_indices: list[int] | None = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.freeze_backbone = freeze_backbone
        self.use_lora = use_lora
        self.use_dpt_fpn = use_dpt_fpn

        # LoRA injection (before freezing so LoRA params stay trainable)
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
                lora_layers=lora_block_indices,
            )
            if self.lora_layers == 0:
                raise RuntimeError(f"No DINOv3 Linear layers matched LoRA targets: {lora_targets}")

        # Freeze backbone (LoRA params remain trainable)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        if use_dpt_fpn:
            # Multi-layer encoder → DPT neck → FPN decoder
            self.encoder = DinoMultiLayerEncoder(
                backbone=self.backbone,
                extract_layers=extract_layers,
                freeze_backbone=freeze_backbone,
                use_lora=use_lora,
            )
            self.neck = DPTReassembleNeck(
                in_ch=DINOV3_FEATURE_DIM,
                out_ch=neck_channels,
                layers=extract_layers,
            )
            self.decoder = FPNDecoder(channels=neck_channels)
            self.extract_layers = list(extract_layers)
        else:
            # Single-layer encoder → SimpleSegHead (DINOv3-IML style)
            self.encoder = DinoSingleLayerEncoder(
                backbone=self.backbone,
                last_layer_idx=last_layer_idx,
                freeze_backbone=freeze_backbone,
                use_lora=use_lora,
            )
            self.decoder = SimpleSegHead(feat_dim=DINOV3_FEATURE_DIM)
            self.neck = None
            self.extract_layers = []

    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        if clip.ndim != 5:
            raise ValueError(f"clip must have shape [B, T, 3, H, W], got {tuple(clip.shape)}")

        batch_size, num_frames, channels, height, width = clip.shape
        if channels != 3:
            raise ValueError(f"clip channel dimension must be 3, got {channels}")

        if self.use_dpt_fpn:
            # Multi-layer encoder → {layer: [B, T, 1024, 32, 32]}
            multi_feats = self.encoder(clip)

            # Decode per-frame through neck + FPN
            logits_list: list[torch.Tensor] = []
            for t in range(num_frames):
                # Gather features for frame t: {layer: [B, 1024, 32, 32]}
                frame_feats = {k: v[:, t] for k, v in multi_feats.items()}
                pyramid = self.neck(frame_feats)  # {"p2": [B,256,128,128], ...}
                logits = self.decoder(pyramid)     # [B, 1, 512, 512]
                logits_list.append(logits)
        else:
            # Single-layer encoder → [B, T, 1024, H/16, W/16]
            encoder_features = self.encoder(clip)

            logits_list = []
            for t in range(num_frames):
                enc_t = encoder_features[:, t]  # [B, 1024, h, w]
                logits = self.decoder(enc_t)    # [B, 1, H, W]
                logits_list.append(logits)

        return torch.stack(logits_list, dim=1)  # [B, T, 1, H, W]


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

    # GPU ID 范围检查
    gpu_id = int(cfg.get("gpu_id", 0))
    if gpu_id < 0:
        raise ValueError(f"gpu_id must be >= 0, got {gpu_id}")
    if torch.cuda.is_available() and gpu_id >= torch.cuda.device_count():
        print(f"[warning] gpu_id={gpu_id} >= available devices ({torch.cuda.device_count()}), "
              f"fallback to cuda:0 or CUDA_VISIBLE_DEVICES")


def build_model(cfg: dict[str, Any], device: torch.device, base_dir: Path) -> DINOv3ViTL16InpaintingDetector:
    backbone = load_dinov3_backbone(cfg, base_dir)
    lora_targets = tuple(str(item).strip() for item in str(cfg.get("lora_targets", "attn.qkv,attn.proj")).split(",") if str(item).strip())

    use_dpt_fpn = bool(cfg.get("use_dpt_fpn", False))
    extract_layers_str = cfg.get("extract_layers", "5,11,17,23")
    if isinstance(extract_layers_str, (list, tuple)):
        extract_layers = tuple(int(x) for x in extract_layers_str)
    else:
        extract_layers = tuple(int(x.strip()) for x in str(extract_layers_str).split(",") if x.strip())
    neck_channels = int(cfg.get("neck_channels", 256))
    lora_block_indices = parse_lora_layers(cfg.get("lora_layers", "all"))

    model = DINOv3ViTL16InpaintingDetector(
        backbone=backbone,
        freeze_backbone=bool(cfg.get("freeze_backbone", True)),
        use_lora=bool(cfg.get("use_lora", False)),
        lora_rank=int(cfg.get("lora_rank", 4)),
        lora_alpha=float(cfg.get("lora_alpha", 8.0)),
        lora_dropout=float(cfg.get("lora_dropout", 0.0)),
        lora_targets=lora_targets,
        use_dpt_fpn=use_dpt_fpn,
        extract_layers=extract_layers,
        neck_channels=neck_channels,
        lora_block_indices=lora_block_indices,
    )

    checkpoint = cfg.get("checkpoint", "")
    if checkpoint:
        state = torch.load(checkpoint, map_location="cpu")
        state_dict = state["model"] if isinstance(state, dict) and "model" in state else state
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[checkpoint] missing keys ({len(missing)}): "
                  f"{missing[:5]}{'...' if len(missing) > 5 else ''}")
        if unexpected:
            print(f"[checkpoint] unexpected keys ({len(unexpected)}): "
                  f"{unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
        print(f"[checkpoint] loaded: {checkpoint}")

    total, trainable = count_parameters(model)
    tag = "DPT+FPN" if use_dpt_fpn else "single-layer"
    print(f"[model] DINOv3 ViT-L/16 {tag} encoder, LoRA layers: {model.lora_layers}")
    print(f"[params] total {total:,} trainable {trainable:,}")
    return model.to(device)


# ── Helpers: logging (dynamic loss keys) ────────────────────────────────────

METRIC_KEYS = ["iou", "f1", "precision", "recall", "accuracy"]


def build_log_fields(active_loss_names: list[str]) -> list[str]:
    """Build LOG_FIELDS list dynamically from active loss names."""
    fields = ["epoch", "lr", "train_loss"]
    for name in active_loss_names:
        fields.append(f"train_{name}_loss")
    for key in METRIC_KEYS:
        fields.append(f"train_{key}")
    fields.append("val_loss")
    for name in active_loss_names:
        fields.append(f"val_{name}_loss")
    for key in METRIC_KEYS:
        fields.append(f"val_{key}")
    return fields


def build_meters(active_loss_names: list[str]) -> dict[str, AverageMeter]:
    """Build meters dict dynamically from active loss names."""
    meters = {"loss": AverageMeter()}
    for name in active_loss_names:
        meters[f"{name}_loss"] = AverageMeter()
    for key in METRIC_KEYS:
        meters[key] = AverageMeter()
    return meters


def summarize_samples(value: Any) -> str:
    paths = split_path_list(value)
    return ";".join(Path(path).name for path in paths)


def init_training_log(path: Path, cfg: dict[str, Any], model: nn.Module, criterion: SegmentationLoss) -> None:
    total, trainable = count_parameters(model)
    log_fields = build_log_fields(criterion.active_names)
    loss_summary = " + ".join(f"{cfg.get('loss', {}).get(name, {}).get('weight', 1.0):.1f}*{name}" for name in criterion.active_names)
    lines = [
        f"# model={DINOV3_MODEL_NAME} use_lora={bool(cfg.get('use_lora', False))} "
        f"lora_rank={int(cfg.get('lora_rank', 4))} input_size={int(cfg.get('input_size', 0))} "
        f"num_frames={int(cfg.get('num_frames', 0))} loss={loss_summary}",
        f"# train_samples={summarize_samples(cfg.get('train_samples', ''))} "
        f"val_samples={summarize_samples(cfg.get('val_samples', ''))}",
        f"# batch_size={int(cfg.get('batch_size', 0))} lr={float(cfg.get('learning_rate', 0.0))} "
        f"params_total={total} params_trainable={trainable}",
        ",".join(log_fields),
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
    log_fields: list[str],
    val_metrics: dict[str, float] | None = None,
) -> None:
    row_values: dict[str, float | int | None] = {"epoch": epoch, "lr": lr}
    for key in ["loss"] + METRIC_KEYS + [name for name in train_metrics if name.endswith("_loss") and name != "loss"]:
        row_values[f"train_{key}"] = train_metrics.get(key)
        row_values[f"val_{key}"] = val_metrics.get(key) if val_metrics is not None else None

    with open(path, "a", encoding="utf-8") as f:
        f.write(",".join(format_log_value(row_values.get(field)) for field in log_fields) + "\n")


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

    meters = build_meters(criterion.active_names)

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
    criterion = SegmentationLoss(loss_cfg=cfg.get("loss", {}))
    log_fields = build_log_fields(criterion.active_names)
    log_path = save_dir / "log.txt"

    # ── resume detection ─────────────────────────────────────────────────
    checkpoint_path = cfg.get("checkpoint", "")
    resume_epoch = 0
    best_iou = -1.0

    if checkpoint_path and Path(checkpoint_path).exists():
        state = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(state, dict):
            resume_epoch = state.get("epoch", 0)
            best_iou = state.get("metrics", {}).get("iou", -1.0)

    if log_path.exists() and resume_epoch > 0:
        print(f"[log] resuming — appending to {log_path} (epoch {resume_epoch}, best_iou={best_iou:.4f})")
    else:
        init_training_log(log_path, cfg, model, criterion)
        print(f"[log] writing training metrics to {log_path}")
    print(f"[loss] active: {criterion.active_names}")
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters. Enable decoder training or LoRA.")

    optimizer = Adam(
        trainable_params,
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
        betas=(0.9, 0.999),
    )
    # ── LR scheduler ──────────────────────────────────────────────────────
    scheduler_type = str(cfg.get("scheduler", "cosine")).strip().lower()
    max_epochs = int(cfg["n_epochs"])
    warmup_epochs = int(cfg.get("warmup_epochs", 10))
    min_lr = float(cfg.get("min_lr", 1e-6))

    if scheduler_type == "plateau":
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=int(cfg.get("plateau_patience", 5)),
            threshold=float(cfg.get("plateau_threshold", 0.003)),
            threshold_mode="abs",
            cooldown=int(cfg.get("plateau_cooldown", 1)),
            min_lr=min_lr,
        )
        scheduler_step_on_val = True   # plateau needs val_iou

    elif scheduler_type == "cosine":
        warmup_scheduler = LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs,
        )
        cosine_scheduler = CosineAnnealingLR(
            optimizer, T_max=max_epochs - warmup_epochs, eta_min=min_lr,
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs],
        )
        scheduler_step_on_val = False  # cosine steps every epoch, no val_iou

    else:
        raise ValueError(f"Unknown scheduler type '{scheduler_type}'. Use 'cosine' or 'plateau'.")

    print(f"[scheduler] type={scheduler_type} warmup_epochs={warmup_epochs} min_lr={min_lr}")
    if scheduler_type == "plateau":
        print(f"[scheduler] plateau: patience={int(cfg.get('plateau_patience', 5))} "
              f"threshold={float(cfg.get('plateau_threshold', 0.003))} "
              f"cooldown={int(cfg.get('plateau_cooldown', 1))}")

    # ── restore optimizer & scheduler state from checkpoint ──────────────
    if checkpoint_path and Path(checkpoint_path).exists() and resume_epoch > 0:
        state = torch.load(checkpoint_path, map_location=device)
        if isinstance(state, dict) and "optimizer" in state:
            optimizer.load_state_dict(state["optimizer"])
            print(f"[resume] optimizer state restored (epoch {resume_epoch})")
        if isinstance(state, dict) and "scheduler" in state:
            scheduler.load_state_dict(state["scheduler"])
            print(f"[resume] scheduler state restored")

    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg.get("amp", True)) and device.type == "cuda")

    train_loader = make_loader(cfg, "train")
    val_loader = make_loader(cfg, "val")

    start_epoch = resume_epoch + 1
    for epoch in range(start_epoch, int(cfg["n_epochs"]) + 1):
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
            # grad_accum_steps=int(cfg.get("grad_accum_steps", 1)),
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
                    # eval_frame_chunk=int(cfg.get("eval_frame_chunk", 1)),
                )
            print(
                f"[epoch {epoch}] val loss {val_metrics['loss']:.4f} "
                f"f1 {val_metrics['f1']:.4f} iou {val_metrics['iou']:.4f}"
            )

            # Save checkpoint with scheduler state
            checkpoint_dict = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "metrics": val_metrics,
            }
            torch.save(checkpoint_dict, save_dir / "latest.pt")

            if val_metrics["iou"] > best_iou:
                best_iou = val_metrics["iou"]
                torch.save(checkpoint_dict, save_dir / "best_iou.pt")
                print(f"[checkpoint] best_iou updated: {best_iou:.4f}")

            # Plateau scheduler: step with val_iou (only on validation epochs)
            if scheduler_step_on_val:
                scheduler.step(val_metrics["iou"])

        # Cosine scheduler: step every epoch without val_iou
        if not scheduler_step_on_val:
            scheduler.step()

        append_epoch_log(log_path, epoch, optimizer.param_groups[0]["lr"], train_metrics, log_fields, val_metrics)


def evaluate(cfg: dict[str, Any], device: torch.device, mode: str, base_dir: Path, *, test_subset: str = "") -> dict[str, float]:
    model = build_model(cfg, device, base_dir)
    criterion = SegmentationLoss(loss_cfg=cfg.get("loss", {}))
    loader = make_loader(cfg, mode)
    vis_dir = Path(cfg["visualization_dir"]) / mode if cfg.get("visualization_dir") else None
    if test_subset and vis_dir:
        vis_dir = vis_dir / test_subset

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
    tag = f"[{mode}]" if not test_subset else f"[{mode}/{test_subset}]"
    print(
        f"{tag} loss {metrics['loss']:.4f} "
        f"f1 {metrics['f1']:.4f} iou {metrics['iou']:.4f} "
        f"precision {metrics['precision']:.4f} recall {metrics['recall']:.4f}"
    )
    return metrics


# ── predefined test-suite subsets ───────────────────────────────────────

TEST_SUITE = [
    {"key": "DVI_20",  "samples": "./flist/DAVIS-VI_val_DVI_20.npy"},
    {"key": "CPNET_20", "samples": "./flist/DAVIS-VI_val_CPNET_20.npy"},
    {"key": "OPN_20",  "samples": "./flist/DAVIS-VI_val_OPN_20.npy"},
]


def print_test_summary(
    results: list[dict[str, Any]],
    cfg: dict[str, Any],
    checkpoint: str,
) -> None:
    """Print a formatted summary after running all test subsets."""

    # ---- gather config info -----------------------------------------------
    train_info = [Path(p).stem for p in split_path_list(cfg.get("train_samples", ""))]
    decoder = "DPT+FPN" if bool(cfg.get("use_dpt_fpn", False)) else "ProgressiveDecoder"
    lora_info = (
        f"rank={int(cfg.get('lora_rank', 0))} alpha={float(cfg.get('lora_alpha', 0)):.0f}"
        if bool(cfg.get("use_lora", False)) else "disabled"
    )
    loss_names = list(cfg.get("loss", {}).keys())
    loss_str = " + ".join(
        f"{cfg['loss'][n].get('weight', 1.0):.1f}*{n}" for n in loss_names if n != "loss"
    ) or "default"

    lines = []
    lines.append("=" * 68)
    lines.append("                     TEST SUITE SUMMARY")
    lines.append("=" * 68)
    lines.append(f"  Checkpoint       : {checkpoint}")
    lines.append(f"  Decoder          : {decoder}")
    lines.append(f"  LoRA             : {lora_info}")
    lines.append(f"  DINO extract     : {cfg.get('extract_layers', '23')}")
    lines.append(f"  Loss             : {loss_str}")
    lines.append(f"  Train datasets   : {', '.join(train_info)}")
    lines.append("-" * 68)

    # ---- result table ----------------------------------------------------
    header = f"  {'Test Set':<14s} {'IoU':>8s} {'F1':>8s} {'Precision':>10s} {'Recall':>8s} {'Loss':>8s}"
    lines.append(header)
    lines.append("  " + "-" * 58)

    best_iou, best_name = -1.0, ""
    for r in results:
        name = r["subset"]
        m = r["metrics"]
        lines.append(
            f"  {name:<14s} {m['iou']:8.4f} {m['f1']:8.4f} "
            f"{m['precision']:10.4f} {m['recall']:8.4f} {m['loss']:8.4f}"
        )
        if m["iou"] > best_iou:
            best_iou, best_name = m["iou"], name

    lines.append("  " + "-" * 58)
    avg_iou = sum(r["metrics"]["iou"] for r in results) / len(results)
    avg_f1 = sum(r["metrics"]["f1"] for r in results) / len(results)
    lines.append(f"  {'Average':<14s} {avg_iou:8.4f} {avg_f1:8.4f}")
    lines.append(f"  Best: {best_name}  IoU={best_iou:.4f}")
    lines.append("=" * 68)

    print("\n".join(lines))


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
    parser.add_argument("--freeze_backbone", type=str2bool, default=None)
    parser.add_argument("--use_lora", type=str2bool, default=None)
    parser.add_argument("--lora_rank", type=int, default=None)
    parser.add_argument("--lora_alpha", type=float, default=None)
    parser.add_argument("--lora_dropout", type=float, default=None)
    parser.add_argument("--lora_targets", type=str, default=None)
    parser.add_argument("--lora_layers", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--visualization_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--gpu_id", type=int, default=None)
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
        "lora_layers": "all",
        "save_dir": "runs/dinov3_vitl16",
        "visualization_dir": "runs/dinov3_vitl16/vis",
        "checkpoint": "",
        "grad_accum_steps": 1,
        "eval_frame_chunk": 1,
        "gpu_id": 0,
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
    device_str = cfg.get("device", "")
    gpu_id = int(cfg.get("gpu_id", 0))
    if not device_str:
        if torch.cuda.is_available():
            device_str = f"cuda:{gpu_id}"
        else:
            device_str = "cpu"
    device = torch.device(device_str)
    print(f"[device] {device} (gpu_id={gpu_id})")
    print(f"[mode] {mode}")
    print(f"[dinov3] model={DINOV3_MODEL_NAME} weights={cfg['dinov3_weights']}")
    decoder_tag = "DPT+FPN" if bool(cfg.get("use_dpt_fpn", False)) else "single-layer ProgressiveDecoder"
    print(f"[dinov3] encoder: {decoder_tag}")
    print(f"[lora] enabled={bool(cfg.get('use_lora', False))} rank={int(cfg.get('lora_rank', 4))}")

    if mode == "train":
        train(cfg, device, base_dir)
    elif mode == "test":
        # Run all predefined test subsets sequentially
        saved_test_samples = cfg.get("test_samples", None)
        checkpoint = cfg.get("checkpoint", "")

        results: list[dict[str, Any]] = []
        for subset in TEST_SUITE:
            # Replace test_samples for this subset
            cfg["test_samples"] = subset["samples"]
            print(f"\n{'─'*50}")
            print(f"[test] running subset: {subset['key']}  ({subset['samples']})")
            print(f"{'─'*50}")
            metrics = evaluate(cfg, device, "test", base_dir, test_subset=subset["key"])
            results.append({"subset": subset["key"], "metrics": dict(metrics)})

        # Restore original test_samples (if any)
        if saved_test_samples is not None:
            cfg["test_samples"] = saved_test_samples

        print_test_summary(results, cfg, checkpoint)
    else:
        evaluate(cfg, device, mode, base_dir)


if __name__ == "__main__":
    main()
