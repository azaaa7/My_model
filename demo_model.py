from __future__ import annotations

import torch

from my_model import SimpleHRNetInpaintingDetector


def main() -> None:
    model = SimpleHRNetInpaintingDetector()
    model.eval()

    clip = torch.randn(2, 1, 3, 256, 256)
    with torch.no_grad():
        logits = model(clip)

    print("input clip shape:", tuple(clip.shape))
    print("predicted mask logits shape:", tuple(logits.shape))
    assert logits.shape == (2, 1, 1, 256, 256)
    print("model demo: OK")


if __name__ == "__main__":
    main()
