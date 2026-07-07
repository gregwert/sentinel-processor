"""
Reference-based radiometric normalisation for Sentinel-2 imagery.

Supports histogram matching (CDF-based) and linear (mean/std) normalisation.
Both functions operate on stretched uint8 arrays and use only numpy —
no scikit-image dependency — ensuring they are picklable for use inside
ProcessPoolExecutor workers.
"""
from __future__ import annotations

import numpy as np


def compute_reference_stats(images: list[np.ndarray]) -> dict:
    """Compute per-band CDFs and statistics from a list of reference images.

    Args:
        images (list of np.ndarray): Non-empty list of uint8 (H, W, 3) arrays. Each array
            must already be percentile-stretched using the same band indices and stretch
            settings as the target image.

    Returns:
        dict: Dictionary with keys:
            n (int): Number of reference images averaged.
            cdfs (list of 3 np.ndarray, each shape (256,), dtype float32): Per-band
                normalised cumulative distribution functions averaged across all reference
                images. Index 0 = red, 1 = green, 2 = blue.
            mean (np.ndarray, shape (3,), dtype float32): Per-band mean averaged across
                all reference images.
            std (np.ndarray, shape (3,), dtype float32): Per-band standard deviation
                averaged across all reference images.

    Raises:
        ValueError: When images is empty.
    """
    if not images:
        raise ValueError("images must be non-empty")

    n = len(images)
    acc_cdfs = [np.zeros(256, dtype=np.float32) for _ in range(3)]
    acc_mean = np.zeros(3, dtype=np.float32)
    acc_std  = np.zeros(3, dtype=np.float32)

    for img in images:
        for c in range(3):
            channel = img[:, :, c].flatten()
            hist, _ = np.histogram(channel, bins=256, range=(0, 256))
            cdf = hist.cumsum().astype(np.float32)
            cdf /= (cdf[-1] + 1e-6)
            acc_cdfs[c] += cdf
            acc_mean[c] += channel.mean()
            acc_std[c]  += channel.std()

    return {
        "n":    n,
        "cdfs": [acc_cdfs[c] / n for c in range(3)],
        "mean": acc_mean / n,
        "std":  acc_std  / n,
    }


def apply_reference_normalisation(
    img_uint8: np.ndarray,
    reference_stats: dict,
    method: str,
) -> np.ndarray:
    """Normalise img_uint8 to match statistics from cloudless reference images.

    Args:
        img_uint8 (np.ndarray): Shape (H, W, 3), dtype uint8. Target image to normalise.
        reference_stats (dict): Pre-computed statistics as returned by
            compute_reference_stats.
        method (str): One of:
            ``"histogram"`` — shifts the full per-band histogram to match the reference
            CDF. Handles non-linear differences between acquisitions.
            ``"linear"`` — per-band linear rescaling using mean and std. Simpler and more
            robust when averaging across multiple reference images.

    Returns:
        np.ndarray: Shape (H, W, 3), dtype uint8. Normalised image clipped to [0, 255].

    Raises:
        ValueError: When method is not one of the supported values.
    """
    if method not in ("histogram", "linear"):
        raise ValueError(f"method must be 'histogram' or 'linear', got {method!r}")

    result = np.empty_like(img_uint8)

    if method == "histogram":
        for c in range(3):
            channel = img_uint8[:, :, c]
            # Build target CDF
            hist, _ = np.histogram(channel.flatten(), bins=256, range=(0, 256))
            target_cdf = hist.cumsum().astype(np.float32)
            target_cdf /= (target_cdf[-1] + 1e-6)
            # Map each target CDF value to the closest reference CDF value
            ref_cdf = reference_stats["cdfs"][c]
            lut = np.clip(np.searchsorted(ref_cdf, target_cdf), 0, 255).astype(np.uint8)
            result[:, :, c] = lut[channel]

    else:  # linear
        img_f = img_uint8.astype(np.float32)
        for c in range(3):
            src_mean = img_f[:, :, c].mean()
            src_std  = img_f[:, :, c].std() + 1e-6
            ref_mean = float(reference_stats["mean"][c])
            ref_std  = float(reference_stats["std"][c])
            if ref_std < 1e-6:
                result[:, :, c] = img_uint8[:, :, c]
            else:
                scaled = (img_f[:, :, c] - src_mean) / src_std * ref_std + ref_mean
                result[:, :, c] = np.clip(scaled, 0, 255).astype(np.uint8)

    return result
