"""
processing/preprocess.py

Sentinel-2 GeoTIFF ingestion and radiometric pre-processing.
Handles reading multi-band 16-bit imagery via rasterio, selecting RGB
band indices, and performing percentile-based contrast stretching to
produce display-ready uint8 arrays while preserving geospatial metadata.
"""

from typing import Tuple

import numpy as np
import rasterio


def read_sentinel_tiff(path: str, band_indices=(1, 2, 3)) -> Tuple[np.ndarray, dict]:
    """Read a multi-band Sentinel-2 GeoTIFF and return a uint16 array and metadata.

    Args:
        path (str): Filesystem path to a GeoTIFF file readable by rasterio.
        band_indices (tuple of int, optional): 1-based rasterio band indices to read,
            e.g. ``(1, 2, 3)`` for RGB. All indices must be <= the number of bands in
            the file; default (1, 2, 3).

    Returns:
        arr (np.ndarray): Shape (H, W, C), dtype uint16. Pixel values in sensor-native
            DN units, one slice per requested band in the order given by ``band_indices``.
        meta (dict): rasterio metadata dict with CRS, transform, and band count updated
            to match the selected bands.

    Raises:
        ValueError: If any requested band index exceeds the number of bands present in
            the file.
    """
    with rasterio.open(path) as src:
        if src.count < max(band_indices):
            raise ValueError(
                f"File has {src.count} band(s) but band index {max(band_indices)} was requested. "
                f"band_indices must not exceed the number of bands in the file."
            )
        # rasterio.read expects 1-based indices; passing a list reads multiple bands
        data = src.read(list(band_indices))
        arr = np.moveaxis(data, 0, -1).astype(np.uint16)
        meta = src.meta.copy()
        meta.update(count=len(band_indices))
    return arr, meta


def percentile_stretch(arr: np.ndarray, p_low: float = 2.0, p_high: float = 98.0, per_band: bool = True) -> np.ndarray:
    """Map a uint16 (H, W, C) array to uint8 via percentile-based contrast stretching.

    Args:
        arr (np.ndarray): Shape (H, W, C), dtype uint16. Input imagery in sensor DN units.
        p_low (float, optional): Lower percentile used as the clip minimum. Pixels at or
            below this percentile are mapped to 0; default 2.0.
        p_high (float, optional): Upper percentile used as the clip maximum. Pixels at or
            above this percentile are mapped to 255; default 98.0.
        per_band (bool, optional): If True, percentiles are computed independently for each
            band so that each channel fills the full 0-255 range. If False, a single pair
            of percentiles is computed across all bands jointly; default True.

    Returns:
        np.ndarray: Shape (H, W, C), dtype uint8. Contrast-stretched image ready for
            display or downstream uint8 processing.
    """
    h, w, c = arr.shape
    out = np.zeros((h, w, c), dtype=np.uint8)

    if per_band:
        for i in range(c):
            band = arr[:, :, i].astype(np.float32)
            lo = np.percentile(band, p_low)
            hi = np.percentile(band, p_high)
            if hi == lo:
                out[:, :, i] = 0
                continue
            stretched = np.clip((band - lo) / (hi - lo + 1e-6), 0.0, 1.0)
            out[:, :, i] = (stretched * 255).astype(np.uint8)
    else:
        flat = arr.astype(np.float32)
        lo = np.percentile(flat, p_low)
        hi = np.percentile(flat, p_high)
        if hi == lo:
            return out
        for i in range(c):
            band = flat[:, :, i]
            stretched = np.clip((band - lo) / (hi - lo + 1e-6), 0.0, 1.0)
            out[:, :, i] = (stretched * 255).astype(np.uint8)

    return out
