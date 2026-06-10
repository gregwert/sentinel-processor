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
    """Read 16-bit Sentinel TIFF; return (uint16 HWC ndarray, rasterio meta dict)."""
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


def percentile_stretch(arr: np.ndarray, p_low=2.0, p_high=98.0, per_band=True) -> np.ndarray:
    """Stretch uint16 (H,W,C) to uint8 using percentile clipping."""
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
