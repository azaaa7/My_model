from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = 1e-7

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits).clamp(self.eps, 1.0 - self.eps)
        alpha = torch.where(
            target == 1,
            torch.full_like(probs, self.alpha),
            torch.full_like(probs, 1.0 - self.alpha),
        )
        pt = torch.where(target == 1, probs, 1.0 - probs)
        ce_loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        loss = alpha * torch.pow(1.0 - pt + self.eps, self.gamma) * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class IoULoss(nn.Module):
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        intersection = (pred * target).sum(dim=(2, 3))
        union = (pred + target).sum(dim=(2, 3)) - intersection
        iou = (intersection + self.smooth) / (union + self.smooth)
        return 1.0 - iou.mean()


class TverskyLoss(nn.Module):
    """Tversky Loss — generalised Dice with asymmetric FP/FN penalty.

    Tversky index:
        TI = (TP + smooth) / (TP + α·FP + β·FN + smooth)

    Loss = 1 - TI

    Args:
        alpha: weight for false positives.  higher → penalise FP → improve precision.
        beta:  weight for false negatives.  higher → penalise FN → improve recall.
        smooth: numerical stabiliser.

    α = β = 0.5  → equivalent to Dice Loss.
    α > β        → focuses on reducing false positives.
    α < β        → focuses on reducing false negatives.
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        logits: [B, 1, H, W]
        target: [B, 1, H, W]
        """
        probs = torch.sigmoid(logits)

        probs = probs.flatten(1)
        target = target.flatten(1)

        tp = (probs * target).sum(dim=1)
        fp = (probs * (1.0 - target)).sum(dim=1)
        fn = ((1.0 - probs) * target).sum(dim=1)

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1.0 - tversky.mean()


class EdgeLoss(nn.Module):
    """Edge-aware BCE loss — weight boundary pixels higher (DINOv3-IML style).

    Computes an edge mask from the ground-truth via morphological gradient
    (dilation − erosion), then applies a weighted BCE where edge pixels are
    scaled by ``edge_lambda`` (paper default 20.0).

    Args:
        edge_lambda: multiplier for edge-region BCE.  Higher → emphasise
            boundary accuracy.  0 → equivalent to plain BCE.
        kernel_size: morphological kernel size for edge extraction.
    """

    def __init__(self, edge_lambda: float = 20.0, kernel_size: int = 3):
        super().__init__()
        self.edge_lambda = edge_lambda
        self.kernel_size = kernel_size

    @staticmethod
    def _compute_edge_mask(target: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
        """Morphological gradient: maxpool − minpool ≈ dilation − erosion."""
        pad = kernel_size // 2
        dilated = F.max_pool2d(target, kernel_size, stride=1, padding=pad)
        eroded = -F.max_pool2d(-target, kernel_size, stride=1, padding=pad)
        edge = (dilated - eroded).clamp(0, 1)
        return edge.detach()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Weighted BCE with edge emphasis.

        Args:
            logits: [B, 1, H, W]
            target: [B, 1, H, W]  binary mask
        Returns:
            scalar loss
        """
        edge_mask = self._compute_edge_mask(target, self.kernel_size)
        weight = 1.0 + self.edge_lambda * edge_mask
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        return (bce * weight).mean()


# class SegmentationLoss(nn.Module):
#     """ZZZ_model-style loss: focal + BCE-with-logits + IoU loss."""

#     def __init__(self):
#         super().__init__()
#         self.focal = FocalLoss()
#         self.iou = IoULoss()

#     def forward(self, logits: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
#         target = target.float()
#         probs = torch.sigmoid(logits)
#         focal_loss = self.focal(logits.view(logits.size(0), -1), target.view(target.size(0), -1))
#         bce_loss = F.binary_cross_entropy_with_logits(
#             logits.view(logits.size(0), -1),
#             target.view(target.size(0), -1),
#         )
#         iou_loss = self.iou(probs, target)
#         loss = focal_loss + bce_loss + iou_loss
#         return loss, {
#             "loss": float(loss.detach().cpu()),
#             "focal_loss": float(focal_loss.detach().cpu()),
#             "bce_loss": float(bce_loss.detach().cpu()),
#             "iou_loss": float(iou_loss.detach().cpu()),
#         }
class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        logits: [B, 1, H, W]
        target: [B, 1, H, W]
        """
        probs = torch.sigmoid(logits)

        probs = probs.flatten(1)
        target = target.flatten(1)

        intersection = (probs * target).sum(dim=1)
        union = probs.sum(dim=1) + target.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class WeightedBCELoss(nn.Module):
    def __init__(self, max_pos_weight: float = 20.0):
        super().__init__()
        self.max_pos_weight = max_pos_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        logits: [B, 1, H, W]
        target: [B, 1, H, W]
        """
        target = target.float()

        pos = target.sum()
        neg = target.numel() - pos

        pos_weight = neg / (pos + 1.0)
        pos_weight = pos_weight.clamp(min=1.0, max=self.max_pos_weight).detach()

        bce = F.binary_cross_entropy_with_logits(
            logits,
            target,
            reduction="none"
        )

        weight = torch.where(
            target == 1,
            torch.full_like(target, pos_weight),
            torch.ones_like(target)
        )

        loss = bce * weight
        return loss.mean()


class SegmentationLoss(nn.Module):
    """Config-driven loss: freely combine WBce / Dice / Focal / IoU / BCE / Tversky.

    Configure via YAML:

        loss:
          wbce:
            weight: 0.5
            max_pos_weight: 20.0
          dice:
            weight: 0.5
            smooth: 1.0e-6
          focal:
            weight: 1.0
            alpha: 0.25
            gamma: 2.0
          iou:
            weight: 1.0
            smooth: 1.0e-6
          bce:
            weight: 1.0
          tversky:
            weight: 1.0
            alpha: 0.3
            beta: 0.7
            smooth: 1.0e-6
          edge:
            weight: 1.0
            edge_lambda: 20.0
            kernel_size: 3

    Any subset is valid.  Weights are normalised by the caller if desired;
    here they are applied as-is so the sum reflects the relative contribution.
    """

    SUPPORTED = {"wbce", "dice", "focal", "iou", "bce", "tversky", "edge"}

    def __init__(self, loss_cfg: dict | None = None):
        super().__init__()
        loss_cfg = loss_cfg or {}

        self.loss_modules: dict[str, nn.Module] = {}
        self.loss_weights: dict[str, float] = {}

        for name, args in loss_cfg.items():
            name = name.strip().lower()
            if name not in self.SUPPORTED:
                raise ValueError(
                    f"Unknown loss '{name}'. Supported: {sorted(self.SUPPORTED)}"
                )
            weight = float(args.get("weight", 1.0))
            self.loss_weights[name] = weight

            if name == "wbce":
                self.loss_modules[name] = WeightedBCELoss(
                    max_pos_weight=float(args.get("max_pos_weight", 20.0))
                )
            elif name == "dice":
                self.loss_modules[name] = DiceLoss(
                    smooth=float(args.get("smooth", 1e-6))
                )
            elif name == "focal":
                self.loss_modules[name] = FocalLoss(
                    alpha=float(args.get("alpha", 0.25)),
                    gamma=float(args.get("gamma", 2.0)),
                )
            elif name == "iou":
                self.loss_modules[name] = IoULoss(
                    smooth=float(args.get("smooth", 1e-6))
                )
            elif name == "tversky":
                self.loss_modules[name] = TverskyLoss(
                    alpha=float(args.get("alpha", 0.3)),
                    beta=float(args.get("beta", 0.7)),
                    smooth=float(args.get("smooth", 1e-6)),
                )
            elif name == "edge":
                self.loss_modules[name] = EdgeLoss(
                    edge_lambda=float(args.get("edge_lambda", 20.0)),
                    kernel_size=int(args.get("kernel_size", 3)),
                )
            elif name == "bce":
                self.loss_modules[name] = None  # marker, handled in forward

    @property
    def active_names(self) -> list[str]:
        """Return ordered list of active loss names for logging / meters."""
        return list(self.loss_weights.keys())

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:

        target = target.float()

        if target.dim() == 3:
            target = target.unsqueeze(1)
        if logits.dim() == 3:
            logits = logits.unsqueeze(1)

        total = torch.tensor(0.0, device=logits.device)
        items: dict[str, float] = {}

        for name, module in self.loss_modules.items():
            weight = self.loss_weights[name]

            if name == "bce":
                val = F.binary_cross_entropy_with_logits(
                    logits.view(logits.size(0), -1),
                    target.view(target.size(0), -1),
                )
            elif name in ("focal",):
                # FocalLoss internally handles sigmoid + BCE
                val = module(logits.view(logits.size(0), -1), target.view(target.size(0), -1))
            elif name == "iou":
                probs = torch.sigmoid(logits)
                val = module(probs, target)
            elif name == "tversky":
                # TverskyLoss internally handles sigmoid
                val = module(logits, target)
            elif name == "edge":
                # EdgeLoss internally handles sigmoid (BCEWithLogits)
                val = module(logits, target)
            else:
                val = module(logits, target)

            total = total + weight * val
            items[f"{name}_loss"] = float(val.detach().cpu())

        items["loss"] = float(total.detach().cpu())
        return total, items