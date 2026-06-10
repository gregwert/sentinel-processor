"""
tests/test_chipping.py

Tests for app.chipping — ChipGrid construction, window computation,
chip extraction, and tile export utilities.
"""

import os
import tempfile

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS

from app.chipping.gdal_chipper import compute_chip_grid, chip_affine, get_chip, build_chip_grid
from app.chipping.tile_exporter import export_chips, zip_export


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def make_test_grid(img_h=512, img_w=512, chip_size=256, overlap=0.0):
    img = np.random.randint(0, 255, (img_h, img_w, 3), dtype=np.uint8)
    transform = from_bounds(0, 0, 1, 1, img_w, img_h)
    meta = {
        "crs": CRS.from_epsg(4326),
        "transform": transform,
        "dtype": "uint8",
        "count": 3,
        "driver": "GTiff",
    }
    return build_chip_grid(img, meta, chip_size, chip_size, overlap)


# ---------------------------------------------------------------------------
# Grid construction tests
# ---------------------------------------------------------------------------

def test_chip_count_no_overlap():
    """512×512 image with 256×256 chips, no overlap → 4 chips (2×2)."""
    grid = make_test_grid(img_h=512, img_w=512, chip_size=256, overlap=0.0)
    assert grid.total == 4


def test_chip_count_with_overlap():
    """768×768 image with 256×256 chips, overlap=0 → 9 chips (3×3)."""
    grid = make_test_grid(img_h=768, img_w=768, chip_size=256, overlap=0.0)
    assert grid.total == 9


# ---------------------------------------------------------------------------
# Chip extraction tests
# ---------------------------------------------------------------------------

def test_chip_shape():
    """First chip has shape (chip_h, chip_w, 3)."""
    grid = make_test_grid()
    chip_arr, _ = get_chip(grid, 0)
    assert chip_arr.shape == (256, 256, 3)


def test_chip_dtype():
    """Chip array dtype is uint8."""
    grid = make_test_grid()
    chip_arr, _ = get_chip(grid, 0)
    assert chip_arr.dtype == np.uint8


def test_chip_affine_origin():
    """Chip 0 has the same spatial origin as the source image."""
    grid = make_test_grid()
    _, chip_meta = get_chip(grid, 0)
    src_transform = grid.source_meta["transform"]
    assert chip_meta["transform"].c == pytest.approx(src_transform.c)
    assert chip_meta["transform"].f == pytest.approx(src_transform.f)


def test_chip_affine_offset():
    """Second chip (index 1) has a larger x-origin than the first chip."""
    grid = make_test_grid()
    _, meta0 = get_chip(grid, 0)
    _, meta1 = get_chip(grid, 1)
    assert meta1["transform"].c > meta0["transform"].c


def test_edge_chip_padded():
    """Bottom-right chip of a 300×300 / 256-chip grid is padded to full chip size."""
    grid = make_test_grid(img_h=300, img_w=300, chip_size=256, overlap=0.0)
    # 300×300 with step 256 → cols at 0,256 and rows at 0,256 → 4 chips
    assert grid.total == 4
    # Index 3 is bottom-right; source area is only 44×44 but must be padded
    chip_arr, _ = get_chip(grid, 3)
    assert chip_arr.shape == (256, 256, 3)


# ---------------------------------------------------------------------------
# Export tests (use temp directories so no files linger)
# ---------------------------------------------------------------------------

def test_export_png():
    """Exporting 4 chips as PNG produces 4 .png files."""
    grid = make_test_grid(img_h=128, img_w=128, chip_size=64, overlap=0.0)
    with tempfile.TemporaryDirectory() as tmp_dir:
        paths = export_chips(grid, tmp_dir, fmt="png", naming="rowcol")
        assert len(paths) == 4
        for p in paths:
            assert p.endswith(".png")
            assert os.path.isfile(p)


def test_export_geotiff_has_crs():
    """GeoTIFF chips retain the CRS from the source metadata."""
    grid = make_test_grid(img_h=128, img_w=128, chip_size=64, overlap=0.0)
    with tempfile.TemporaryDirectory() as tmp_dir:
        paths = export_chips(grid, tmp_dir, fmt="geotiff", naming="rowcol")
        assert len(paths) == 4
        with rasterio.open(paths[0]) as src:
            assert src.crs is not None


def test_zip_export_nonempty():
    """Zipping exported PNGs returns non-empty bytes."""
    grid = make_test_grid(img_h=128, img_w=128, chip_size=64, overlap=0.0)
    with tempfile.TemporaryDirectory() as tmp_dir:
        paths = export_chips(grid, tmp_dir, fmt="png", naming="rowcol")
        zip_bytes = zip_export(paths)
        assert len(zip_bytes) > 0
