from __future__ import annotations

import cv2
import numpy as np
import albumentations as A
from albumentations import ImageCompression


def is_image_file(filename: str) -> bool:
    return filename.lower().endswith(
        (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    )


def threshold_mask(mask: np.ndarray, thres: float = 0.5) -> np.ndarray:
    mask = mask.copy()
    mask[mask <= int(thres * 255)] = 0
    mask[mask > int(thres * 255)] = 255
    return mask


def build_replay_augmenter() -> A.ReplayCompose:
    return A.ReplayCompose([
        A.RandomScale(scale_limit=(-0.8, 0.0), p=0.75),
        A.Rotate(limit=45, p=0.2),
        A.HorizontalFlip(p=0.2),
        A.VerticalFlip(p=0.2),
        A.Transpose(p=0.2),
        A.RandomBrightnessContrast(p=0.2),
        A.RandomToneCurve(p=0.2),
        A.ImageCompression(quality_lower=70, quality_upper=90, compression_type="jpeg", p=0.3),
        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.1),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.1),
        A.MotionBlur(p=0.1),
        A.Downscale(scale_min=0.8, scale_max=0.9, p=0.1),
    ])


def simulate_jpeg_compression_cv2(image: np.ndarray, quality: int = 30) -> np.ndarray:
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    success, encoded_img = cv2.imencode(".jpg", image, encode_param)
    if not success:
        raise ValueError("JPEG 编码失败")
    return cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)


def add_gaussian_noise_snr(image: np.ndarray, snr_db: float = 20) -> np.ndarray:
    image_float = image.astype(np.float32)
    if image.ndim == 3:
        signal_power = np.var(image_float, axis=(0, 1))
    else:
        signal_power = np.var(image_float)

    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear

    if image.ndim == 3:
        noise = np.zeros_like(image_float)
        for c in range(3):
            std = np.sqrt(max(float(noise_power[c]), 0.0))
            noise[:, :, c] = np.random.normal(0, std, image.shape[:2])
    else:
        std = np.sqrt(max(float(noise_power), 0.0))
        noise = np.random.normal(0, std, image.shape)

    noisy_image = image_float + noise
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

