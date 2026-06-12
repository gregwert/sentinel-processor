"""
tests/test_chipping.py

Tests for app.chipping — ChipGrid construction, window computation,
chip extraction, and tile export utilities.
"""

import io
import os
import re
import tempfile
import zipfile

import numpy as np
import pytest
import rasterio
from PIL import Image
from rasterio.transform import from_bounds
from rasterio.crs import CRS

from app.chipping.gdal_chipper import compute_chip_grid, chip_affine, get_chip, build_chip_grid
from app.chipping.tile_exporter import _coords_filename, export_chips, zip_export


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


# ---------------------------------------------------------------------------
# _coords_filename tests
# ---------------------------------------------------------------------------

def _geo_chip_meta(lat, lon):
    """Build a minimal chip_meta dict for a geographic (lat/lon) CRS.

    Args:
        lat (float): Latitude of the chip's top-left corner (transform.f).
        lon (float): Longitude of the chip's top-left corner (transform.c).

    Returns:
        dict: Chip metadata with 'transform' and 'crs' keys.
    """
    transform = from_bounds(lon, lat - 0.01, lon + 0.01, lat, 64, 64)
    return {"transform": transform, "crs": CRS.from_epsg(4326)}


def _projected_chip_meta(north, east):
    """Build a minimal chip_meta dict for a projected (UTM) CRS.

    Args:
        north (float): Northing in metres of the chip's top-left corner.
        east (float): Easting in metres of the chip's top-left corner.

    Returns:
        dict: Chip metadata with 'transform' and 'crs' keys.
    """
    transform = from_bounds(east, north - 100, east + 100, north, 64, 64)
    return {"transform": transform, "crs": CRS.from_epsg(32632)}


def test_coords_filename_geographic_contains_cardinal_and_p():
    """Geographic CRS filename should contain N/S, E/W, and 'p' for decimal points."""
    meta = _geo_chip_meta(lat=51.4823, lon=-0.1034)
    name = _coords_filename(meta, ".png")
    assert "N" in name or "S" in name
    assert "E" in name or "W" in name
    assert "p" in name


def test_coords_filename_projected_contains_integer_coords():
    """Projected CRS filename should contain integer easting/northing (no 'p' separator)."""
    meta = _projected_chip_meta(north=5714000, east=341000)
    name = _coords_filename(meta, ".tif")
    assert "N" in name or "S" in name
    assert "E" in name or "W" in name
    # Projected variant uses integers — no decimal-point replacement (digit-p-digit)
    assert not re.search(r"\dp\d", name)


# ---------------------------------------------------------------------------
# export_chips — rejected_indices and include_rejected
# ---------------------------------------------------------------------------

def test_export_chips_rejected_indices_skipped():
    """Chips whose indices appear in rejected_indices are not written."""
    grid = make_test_grid(img_h=128, img_w=128, chip_size=64, overlap=0.0)
    # Grid has 4 chips; reject chips 0 and 1
    with tempfile.TemporaryDirectory() as tmp_dir:
        paths = export_chips(grid, tmp_dir, fmt="png", naming="rowcol", rejected_indices=[0, 1])
        assert len(paths) == 2
        for p in paths:
            assert os.path.isfile(p)


def test_export_chips_include_rejected_writes_all():
    """When include_rejected=True, all chips are written even if rejected_indices is set."""
    grid = make_test_grid(img_h=128, img_w=128, chip_size=64, overlap=0.0)
    with tempfile.TemporaryDirectory() as tmp_dir:
        paths = export_chips(
            grid, tmp_dir, fmt="png", naming="rowcol",
            rejected_indices=[0, 1], include_rejected=True,
        )
        assert len(paths) == 4


# ---------------------------------------------------------------------------
# export_chips — JPEG and NPY formats
# ---------------------------------------------------------------------------

def test_export_chips_jpeg_extension_and_readable():
    """JPEG export produces .jpg files that are readable as PIL images."""
    grid = make_test_grid(img_h=128, img_w=128, chip_size=64, overlap=0.0)
    with tempfile.TemporaryDirectory() as tmp_dir:
        paths = export_chips(grid, tmp_dir, fmt="jpeg", naming="rowcol")
        assert len(paths) == 4
        for p in paths:
            assert p.endswith(".jpg")
            with Image.open(p) as img:
                assert img.size == (64, 64)


def test_export_chips_npy_extension_and_loadable():
    """NPY export produces .npy files that are loadable with np.load."""
    grid = make_test_grid(img_h=128, img_w=128, chip_size=64, overlap=0.0)
    with tempfile.TemporaryDirectory() as tmp_dir:
        paths = export_chips(grid, tmp_dir, fmt="npy", naming="rowcol")
        assert len(paths) == 4
        for p in paths:
            assert p.endswith(".npy")
            arr = np.load(p)
            assert arr.ndim == 3


# ---------------------------------------------------------------------------
# zip_export — empty list
# ---------------------------------------------------------------------------

def test_zip_export_empty_list_returns_valid_zip():
    """zip_export on an empty path list returns bytes that form a valid (empty) ZIP."""
    result = zip_export([])
    assert isinstance(result, bytes)
    with zipfile.ZipFile(io.BytesIO(result)) as zf:
        assert zf.namelist() == []


# ---------------------------------------------------------------------------
# compute_chip_grid / build_chip_grid gap tests
# ---------------------------------------------------------------------------

def test_compute_chip_grid_overlap_produces_more_chips():
    """overlap=0.5 must produce more chips than the zero-overlap count."""
    no_overlap = compute_chip_grid(256, 256, 64, 64, overlap=0.0)
    with_overlap = compute_chip_grid(256, 256, 64, 64, overlap=0.5)
    assert len(with_overlap) > len(no_overlap)


def test_compute_chip_grid_edge_mode_overlap_full_size_windows():
    """With edge_mode='overlap', every window must be exactly chip_w × chip_h."""
    chip_w, chip_h = 64, 64
    windows = compute_chip_grid(100, 100, chip_w, chip_h, overlap=0.0, edge_mode="overlap")
    for col_off, row_off, w, h in windows:
        assert w == chip_w, f"Window width {w} != {chip_w}"
        assert h == chip_h, f"Window height {h} != {chip_h}"


def test_build_chip_grid_n_cols_n_rows():
    """build_chip_grid n_cols and n_rows match expected grid dimensions."""
    # 256×256 image with 64-pixel chips, no overlap → 4×4 = 16 chips
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    transform = from_bounds(0, 0, 1, 1, 256, 256)
    meta = {"crs": CRS.from_epsg(4326), "transform": transform, "dtype": "uint8", "count": 3}
    grid = build_chip_grid(img, meta, chip_w=64, chip_h=64, overlap=0.0)
    assert grid.n_cols == 4
    assert grid.n_rows == 4
