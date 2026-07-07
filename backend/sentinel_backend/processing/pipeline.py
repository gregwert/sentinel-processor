"""
processing/pipeline.py

End-to-end processing pipeline orchestrator.
Chains preprocess → dehaze → enhance into a single callable, parameterised
by PipelineConfig. Returns a PipelineResult containing the final image,
preserved geospatial metadata, and a dict of intermediate stage images
for UI display and debugging.
"""

from dataclasses import dataclass
import numpy as np

from sentinel_backend.processing.preprocess import read_sentinel_tiff, percentile_stretch
from sentinel_backend.processing.dehazing import Dehazer
from sentinel_backend.processing.enhancement import apply_clahe


@dataclass
class PipelineConfig:
    """Configuration for a single end-to-end pipeline run.

    Attributes:
        band_indices (tuple of int): 1-based rasterio band indices to read from the source
            GeoTIFF; default (1, 2, 3).
        p_low (float): Lower percentile for contrast stretching; default 2.0.
        p_high (float): Upper percentile for contrast stretching; default 98.0.
        per_band_stretch (bool): If True, percentile stretch is applied independently per
            band; default True.
        run_dehaze (bool): Whether to run the DCP dehazing stage; default True.
        patch_size (int): Dark-channel patch size forwarded to :func:`~dehazing.dehaze`;
            default 15.
        omega (float): Haze retention factor forwarded to :func:`~dehazing.dehaze`;
            default 0.95.
        t0 (float): Minimum transmission clamp forwarded to :func:`~dehazing.dehaze`;
            default 0.1.
        use_guided_filter (bool): Whether to refine the transmission map with a guided
            filter; default True.
        mask_clouds (bool): Whether to detect and preserve cloud pixels around the dehazing
            step; default True.
        cloud_brightness_thresh (float): Brightness threshold for cloud detection;
            default 0.75.
        cloud_saturation_thresh (float): Saturation threshold for cloud detection;
            default 0.08.
        enhancement (str): Post-dehazing enhancement mode. ``"clahe"`` applies CLAHE;
            ``"none"`` skips; default ``"clahe"``.
        clahe_clip_limit (float): CLAHE clip limit forwarded to
            :func:`~enhancement.apply_clahe`; default 2.0.
        clahe_tile_grid (tuple of int): CLAHE tile grid size forwarded to
            :func:`~enhancement.apply_clahe`; default (8, 8).
    """

    band_indices: tuple[int, ...] = (1, 2, 3)
    p_low: float = 2.0
    p_high: float = 98.0
    per_band_stretch: bool = True
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
    clahe_tile_grid: tuple[int, int] = (8, 8)


@dataclass
class PipelineResult:
    """Outputs produced by a completed pipeline run.

    Attributes:
        image (np.ndarray): Shape (H, W, 3), dtype uint8. Final enhanced image ready for
            display or export.
        meta (dict): rasterio metadata dict with CRS, transform, and output dtype/driver
            settings preserved.
        stages (dict[str, np.ndarray]): Intermediate uint8 images keyed by stage name
            (``"preprocessed"``, ``"dehazed"``, ``"enhanced"``), used for UI display and
            debugging.
    """

    image: np.ndarray          # (H, W, 3) uint8 — final enhanced image
    meta: dict                 # rasterio metadata with CRS/transform preserved
    stages: dict[str, np.ndarray]  # intermediate images keyed by stage name


def run_pipeline(tiff_path: str, config: PipelineConfig) -> PipelineResult:
    """Orchestrate the preprocess → dehaze → enhance chain and return all stage outputs.

    Args:
        tiff_path (str): Filesystem path to the source Sentinel-2 GeoTIFF.
        config (PipelineConfig): Fully specified run configuration controlling every stage
            of the pipeline.

    Returns:
        PipelineResult: Container holding the final enhanced image, updated rasterio
            metadata, and a ``stages`` dict with intermediate images for each completed
            stage.
    """
    stages = {}

    # Stage 1: Read + stretch
    raw, meta = read_sentinel_tiff(tiff_path, config.band_indices)
    stretched = percentile_stretch(raw, config.p_low, config.p_high, config.per_band_stretch)
    stages["preprocessed"] = stretched.copy()
    current = stretched

    # Stage 2: Dehaze (optional)
    if config.run_dehaze:
        current = Dehazer(
            patch_size=config.patch_size,
            omega=config.omega,
            t0=config.t0,
            use_guided_filter=config.use_guided_filter,
            mask_clouds=config.mask_clouds,
            brightness_thresh=config.cloud_brightness_thresh,
            saturation_thresh=config.cloud_saturation_thresh,
        ).run(current)
        stages["dehazed"] = current.copy()

    # Stage 3: Enhancement — supports "clahe" or "none"
    if config.enhancement == "clahe":
        current = apply_clahe(current, config.clahe_clip_limit, config.clahe_tile_grid)
    stages["enhanced"] = current.copy()

    # Update meta for uint8 output
    meta.update(dtype="uint8", count=len(config.band_indices), driver="GTiff", compress="lzw", predictor=2)

    return PipelineResult(image=current, meta=meta, stages=stages)
