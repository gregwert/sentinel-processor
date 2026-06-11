"""
processing/enhancement.py

Post-dehazing perceptual enhancement for satellite imagery.
Provides CLAHE-based local contrast enhancement (operating in LAB
colour space to avoid hue distortion) and statistical standardisation
to match a target luminance distribution for downstream model ingestion.
"""

import numpy as np
import cv2


def apply_clahe(img_uint8: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    """Enhance local contrast via CLAHE applied to the L channel in LAB colour space.

    Parameters
    ----------
    img_uint8 : np.ndarray
        Shape (H, W, 3), dtype uint8. Input RGB image.
    clip_limit : float, optional
        CLAHE contrast clip limit. Higher values allow stronger contrast
        enhancement but increase noise amplification; default 2.0.
    tile_grid_size : tuple of int, optional
        ``(rows, cols)`` tile grid used by CLAHE. Smaller tiles enhance local
        contrast more aggressively; default (8, 8).

    Returns
    -------
    np.ndarray
        Shape (H, W, 3), dtype uint8. Contrast-enhanced RGB image with hue and
        saturation unchanged.
    """
    lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, b])
    result = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)
    return result


def apply_standardization(img_uint8: np.ndarray, target_mean: float = 127.5, target_std: float = 45.0) -> np.ndarray:
    """Z-score normalise per-channel then rescale to a target mean and standard deviation.

    Parameters
    ----------
    img_uint8 : np.ndarray
        Shape (H, W, 3), dtype uint8. Input RGB image.
    target_mean : float, optional
        Desired mean pixel value in the output; default 127.5.
    target_std : float, optional
        Desired standard deviation of pixel values in the output; default 45.0.

    Returns
    -------
    np.ndarray
        Shape (H, W, 3), dtype uint8. Image rescaled to the requested
        luminance distribution, clipped to [0, 255].
    """
    img_f = img_uint8.astype(np.float32)
    mean = img_f.mean(axis=(0, 1), keepdims=True)
    std = img_f.std(axis=(0, 1), keepdims=True) + 1e-6
    normalized = (img_f - mean) / std
    rescaled = normalized * target_std + target_mean
    return np.clip(rescaled, 0, 255).astype(np.uint8)
