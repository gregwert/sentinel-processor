"""
tests/test_preprocess.py

Tests for app.processing.preprocess — read_sentinel_tiff and percentile_stretch.
"""

import os
import tempfile

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds

from app.processing.preprocess import percentile_stretch, read_sentinel_tiff


def make_fake_tiff(h=64, w=64, bands=3):
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
        return tmp.name, transform


# ---------------------------------------------------------------------------
# percentile_stretch tests
# ---------------------------------------------------------------------------

def test_percentile_stretch_output_dtype():
    arr = np.random.randint(0, 65535, (100, 100, 3), dtype=np.uint16)
    result = percentile_stretch(arr)
    assert result.dtype == np.uint8


def test_percentile_stretch_output_range():
    arr = np.random.randint(0, 65535, (100, 100, 3), dtype=np.uint16)
    result = percentile_stretch(arr)
    assert result.min() >= 0
    assert result.max() <= 255


def test_percentile_stretch_no_nan():
    arr = np.random.randint(0, 65535, (100, 100, 3), dtype=np.uint16)
    result = percentile_stretch(arr).astype(np.float32)
    assert not np.any(np.isnan(result))


def test_percentile_stretch_shape_preserved():
    arr = np.random.randint(0, 65535, (100, 100, 3), dtype=np.uint16)
    result = percentile_stretch(arr)
    assert result.shape == arr.shape


def test_percentile_stretch_flat_band():
    arr = np.random.randint(1, 3000, (100, 100, 3), dtype=np.uint16)
    arr[:, :, 1] = 0  # make band index 1 entirely flat
    result = percentile_stretch(arr)
    assert np.all(result[:, :, 1] == 0)


def test_percentile_stretch_per_band_vs_global():
    rng = np.random.default_rng(42)
    arr = np.zeros((100, 100, 3), dtype=np.uint16)
    # Give each band a very different scale so per-band and global diverge
    arr[:, :, 0] = rng.integers(0, 500, (100, 100), dtype=np.uint16)
    arr[:, :, 1] = rng.integers(1000, 3000, (100, 100), dtype=np.uint16)
    arr[:, :, 2] = rng.integers(5000, 30000, (100, 100), dtype=np.uint16)

    per_band_result = percentile_stretch(arr, per_band=True)
    global_result = percentile_stretch(arr, per_band=False)

    assert not np.array_equal(per_band_result, global_result)


# ---------------------------------------------------------------------------
# read_sentinel_tiff tests
# ---------------------------------------------------------------------------

def test_read_sentinel_tiff_meta_preserved():
    path, expected_transform = make_fake_tiff()
    try:
        arr, meta = read_sentinel_tiff(path)
        assert "crs" in meta
        assert "transform" in meta
        assert meta["transform"] == expected_transform
    finally:
        os.unlink(path)


def test_read_sentinel_tiff_shape():
    path, _ = make_fake_tiff(h=64, w=64, bands=3)
    try:
        arr, meta = read_sentinel_tiff(path)
        assert arr.shape == (64, 64, 3)
    finally:
        os.unlink(path)
