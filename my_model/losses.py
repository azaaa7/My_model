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


class SegmentationLoss(nn.Module):
    """ZZZ_model-style loss: focal + BCE-with-logits + IoU loss."""

    def __init__(self):
        super().__init__()
        self.focal = FocalLoss()
        self.iou = IoULoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        target = target.float()
        probs = torch.sigmoid(logits)
        focal_loss = self.focal(logits.view(logits.size(0), -1), target.view(target.size(0), -1))
        bce_loss = F.binary_cross_entropy_with_logits(
            logits.view(logits.size(0), -1),
            target.view(target.size(0), -1),
        )
        iou_loss = self.iou(probs, target)
        loss = focal_loss + bce_loss + iou_loss
        return loss, {
            "loss": float(loss.detach().cpu()),
            "focal_loss": float(focal_loss.detach().cpu()),
            "bce_loss": float(bce_loss.detach().cpu()),
            "iou_loss": float(iou_loss.detach().cpu()),
        }
