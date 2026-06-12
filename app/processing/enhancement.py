"""
processing/enhancement.py

Post-dehazing perceptual enhancement for satellite imagery.
Provides CLAHE-based local contrast enhancement (operating in LAB colour
space to avoid hue distortion) and Gray World white balance correction.
"""

import numpy as np
import cv2


def apply_clahe(img_uint8: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    """Enhance local contrast via CLAHE applied to the L channel in LAB colour space.

    Args:
        img_uint8 (np.ndarray): Shape (H, W, 3), dtype uint8. Input RGB image.
        clip_limit (float, optional): CLAHE contrast clip limit. Higher values allow stronger
            contrast enhancement but increase noise amplification; default 2.0.
        tile_grid_size (tuple of int, optional): ``(rows, cols)`` tile grid used by CLAHE.
            Smaller tiles enhance local contrast more aggressively; default (8, 8).

    Returns:
        np.ndarray: Shape (H, W, 3), dtype uint8. Contrast-enhanced RGB image with hue and
            saturation unchanged.
    """
    lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, b])
    result = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)
    return result


def apply_gray_world(
    img_uint8: np.ndarray,
    cloud_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Apply Gray World white balance to correct atmospheric colour cast.

    Scales each RGB channel so its mean (over non-cloud pixels when
    cloud_mask is supplied) equals the grand mean of all channel means.
    Cloud pixels are excluded from mean computation but still scaled in output.

    Args:
        img_uint8 (np.ndarray): Shape (H, W, 3), dtype uint8.
        cloud_mask (np.ndarray or None): Shape (H, W), dtype bool. When provided, only
            non-cloud pixels contribute to the per-channel mean computation. Default None.

    Returns:
        np.ndarray: Shape (H, W, 3), dtype uint8.

    Note:
        Channel means near zero (< 1e-6) are left unscaled to avoid blow-out.
        Gray World assumes spectrally balanced land cover and may over-correct
        on all-desert or all-ocean scenes.
    """
    img_f = img_uint8.astype(np.float32)
    valid = ~cloud_mask if cloud_mask is not None else np.ones(img_uint8.shape[:2], dtype=bool)
    means = np.array([
        img_f[:, :, c][valid].mean() if valid.any() else img_f[:, :, c].mean()
        for c in range(3)
    ], dtype=np.float32)
    grand_mean = means.mean()
    scales = np.where(means > 1e-6, grand_mean / means, 1.0).astype(np.float32)
    return np.clip(img_f * scales[np.newaxis, np.newaxis, :], 0, 255).astype(np.uint8)


