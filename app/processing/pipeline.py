"""
processing/pipeline.py

End-to-end processing pipeline orchestrator.
Chains preprocess → dehaze → enhance into a single callable, parameterised
by PipelineConfig. Returns a PipelineResult containing the final image,
preserved geospatial metadata, and a dict of intermediate stage images
for UI display and debugging.
"""

from dataclasses import dataclass, field
from typing import Tuple, Dict
import numpy as np

from .preprocess import read_sentinel_tiff, percentile_stretch
from .dehazing import dehaze
from .enhancement import apply_clahe


@dataclass
class PipelineConfig:
    band_indices: Tuple[int, ...] = (1, 2, 3)
    p_low: float = 2.0
    p_high: float = 98.0
    per_band_stretch: bool = False
    run_dehaze: bool = True
    patch_size: int = 15
    omega: float = 0.95
    t0: float = 0.1
    use_guided_filter: bool = True
    mask_clouds: bool = True
    cloud_brightness_thresh: float = 0.75
    cloud_saturation_thresh: float = 0.08
    enhancement: str = "clahe"
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: Tuple[int, int] = (8, 8)


@dataclass
class PipelineResult:
    image: np.ndarray          # (H, W, 3) uint8 — final enhanced image
    meta: dict                 # rasterio metadata with CRS/transform preserved
    stages: Dict[str, np.ndarray]  # intermediate images keyed by stage name


def run_pipeline(tiff_path: str, config: PipelineConfig) -> PipelineResult:
    """Orchestrate preprocess → dehaze → enhance; return PipelineResult with stage images."""
    stages = {}

    # Stage 1: Read + stretch
    raw, meta = read_sentinel_tiff(tiff_path, config.band_indices)
    stretched = percentile_stretch(raw, config.p_low, config.p_high, config.per_band_stretch)
    stages["preprocessed"] = stretched.copy()
    current = stretched

    # Stage 2: Dehaze (optional)
    if config.run_dehaze:
        current = dehaze(
            current,
            config.patch_size,
            config.omega,
            config.t0,
            config.use_guided_filter,
            config.mask_clouds,
            config.cloud_brightness_thresh,
            config.cloud_saturation_thresh,
        )
        stages["dehazed"] = current.copy()

    # Stage 3: Enhancement — supports "clahe" or "none"
    if config.enhancement == "clahe":
        current = apply_clahe(current, config.clahe_clip_limit, config.clahe_tile_grid)
    stages["enhanced"] = current.copy()

    # Update meta for uint8 output
    meta.update(dtype="uint8", count=3, driver="GTiff", compress="lzw", predictor=2)

    return PipelineResult(image=current, meta=meta, stages=stages)
