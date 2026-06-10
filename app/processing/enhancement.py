"""
processing/enhancement.py

Post-dehazing perceptual enhancement for satellite imagery.
Provides CLAHE-based local contrast enhancement (operating in LAB
colour space to avoid hue distortion) and statistical standardisation
to match a target luminance distribution for downstream model ingestion.
"""

import numpy as np
import cv2


def apply_clahe(img_uint8, clip_limit=2.0, tile_grid_size=(8, 8)):
    """Apply CLAHE on L channel in LAB space to avoid color hue shifts."""
    lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, b])
    result = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)
    return result


def apply_standardization(img_uint8, target_mean=127.5, target_std=45.0):
    """Z-score normalise then rescale to target mean/std distribution."""
    img_f = img_uint8.astype(np.float32)
    mean = img_f.mean(axis=(0, 1), keepdims=True)
    std = img_f.std(axis=(0, 1), keepdims=True) + 1e-6
    normalized = (img_f - mean) / std
    rescaled = normalized * target_std + target_mean
    return np.clip(rescaled, 0, 255).astype(np.uint8)
