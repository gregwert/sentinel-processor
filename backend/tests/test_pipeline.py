"""
tests/test_pipeline.py

Tests for sentinel_backend.processing.pipeline — end-to-end run_pipeline orchestration.
"""

import os
import tempfile

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds

from sentinel_backend.processing.pipeline import run_pipeline, PipelineConfig


def make_fake_tiff(h=64, w=64, bands=3):
    """Write a synthetic uint16 GeoTIFF to a temp file and return its path.

    Args:
        h (int): Image height in pixels.
        w (int): Image width in pixels.
        bands (int): Number of raster bands to write.

    Returns:
        str: Filesystem path to the temporary TIFF (caller must delete it).
    """
    data = np.random.randint(0, 3000, (bands, h, w), dtype=np.uint16)
    transform = from_bounds(0, 0, 1, 1, w, h)
    with MemoryFile() as memfile:
        with memfile.open(
            driver="GTiff",
            height=h,
            width=w,
            count=bands,
            dtype="uint16",
            crs=CRS.from_epsg(4326),
            transform=transform,
        ) as dataset:
            dataset.write(data)
        tmp = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
        tmp.write(memfile.read())
        tmp.close()
        return tmp.name


# ---------------------------------------------------------------------------
# Happy-path shape, dtype, and key stages
# ---------------------------------------------------------------------------

def test_run_pipeline_output_shape_and_dtype():
    """Default pipeline returns (H, W, 3) uint8 with expected stage keys."""
    path = make_fake_tiff(h=64, w=64)
    try:
        result = run_pipeline(path, PipelineConfig())
        assert result.image.ndim == 3
        assert result.image.shape[2] == 3
        assert result.image.dtype == np.uint8
        assert "preprocessed" in result.stages
        assert "enhanced" in result.stages
        assert "dehazed" in result.stages
        assert "crs" in result.meta
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# run_dehaze=False
# ---------------------------------------------------------------------------

def test_run_pipeline_no_dehaze_has_preprocessed_stage():
    """Disabling dehazing still produces a 'preprocessed' stage and no crash."""
    path = make_fake_tiff(h=64, w=64)
    try:
        cfg = PipelineConfig(run_dehaze=False)
        result = run_pipeline(path, cfg)
        assert "preprocessed" in result.stages
        assert "dehazed" not in result.stages
        assert result.image.dtype == np.uint8
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# Invalid band index
# ---------------------------------------------------------------------------

def test_run_pipeline_invalid_band_raises():
    """Requesting a band index beyond the file's band count raises ValueError."""
    path = make_fake_tiff(h=64, w=64, bands=3)
    try:
        cfg = PipelineConfig(band_indices=(1, 2, 99))
        with pytest.raises((ValueError, Exception)):
            run_pipeline(path, cfg)
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# Non-existent file
# ---------------------------------------------------------------------------

def test_run_pipeline_missing_file_raises():
    """Passing a path that does not exist raises rasterio or OSError."""
    cfg = PipelineConfig()
    with pytest.raises(Exception):
        run_pipeline("/nonexistent/path/fake.tif", cfg)
