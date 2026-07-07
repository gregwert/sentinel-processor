"""
tests/test_manifest.py

Tests for sentinel_backend.chipping.manifest — chip_lat_lon_bounds, build_manifest,
and write_manifest_csv.
"""

import csv
import io

import numpy as np
import pytest
from affine import Affine
from rasterio.crs import CRS
from rasterio.transform import from_bounds

from sentinel_backend.chipping.gdal_chipper import build_chip_grid
from sentinel_backend.chipping.manifest import build_manifest, chip_lat_lon_bounds, write_manifest_csv


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

def make_grid_4chip(crs=None):
    """Return a 2-column × 2-row ChipGrid over a 64×64 image with 32×32 chips.

    Args:
        crs: rasterio CRS to embed in the source metadata, or None for a
            grid without spatial reference.

    Returns:
        ChipGrid with 4 windows: (0,0,32,32), (32,0,32,32),
        (0,32,32,32), (32,32,32,32).
    """
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    transform = from_bounds(west=10.0, south=50.0, east=10.064, north=50.064,
                            width=64, height=64)
    meta = {
        "crs": crs,
        "transform": transform,
        "dtype": "uint8",
        "count": 3,
        "driver": "GTiff",
    }
    return build_chip_grid(img, meta, chip_w=32, chip_h=32)


# Convenience grid instances used by multiple tests.
_GRID_GEO = make_grid_4chip(crs=CRS.from_epsg(4326))
_GRID_NOCRS = make_grid_4chip(crs=None)


# ---------------------------------------------------------------------------
# chip_lat_lon_bounds
# ---------------------------------------------------------------------------

class TestChipLatLonBounds:
    """Tests for chip_lat_lon_bounds."""

    def test_crs_none_returns_four_nones(self):
        """When crs is None the function returns (None, None, None, None)."""
        transform = from_bounds(0, 0, 1, 1, 64, 64)
        result = chip_lat_lon_bounds(transform, None, 0, 0, 32, 32)
        assert result == (None, None, None, None)

    @pytest.mark.parametrize("col_off,row_off", [
        (0, 0),
        (32, 0),
        (0, 32),
        (32, 32),
    ])
    def test_geographic_crs_returns_plausible_floats(self, col_off, row_off):
        """Geographic CRS yields floats and lon_min < lon_max, lat_min < lat_max."""
        transform = from_bounds(10.0, 50.0, 10.064, 50.064, 64, 64)
        crs = CRS.from_epsg(4326)
        lon_min, lat_min, lon_max, lat_max = chip_lat_lon_bounds(
            transform, crs, col_off, row_off, 32, 32
        )
        assert all(isinstance(v, float) for v in (lon_min, lat_min, lon_max, lat_max))
        assert lon_min < lon_max
        assert lat_min < lat_max

    def test_top_left_corner_matches_transform_origin(self):
        """For col_off=0, row_off=0 the lon_min equals transform.c exactly."""
        # A simple north-up affine with known origin at (10.0 lon, 50.064 lat)
        # pixel size = 0.001 degrees per pixel
        transform = Affine(0.001, 0.0, 10.0,
                           0.0, -0.001, 50.064)
        crs = CRS.from_epsg(4326)
        lon_min, lat_min, lon_max, lat_max = chip_lat_lon_bounds(
            transform, crs, 0, 0, 32, 32
        )
        # Top-left corner: lon = transform.c, lat_max = transform.f
        assert lon_min == pytest.approx(10.0, abs=1e-9)
        assert lat_max == pytest.approx(50.064, abs=1e-9)
        # Bottom-right corner after 32 pixels
        assert lon_max == pytest.approx(10.0 + 32 * 0.001, abs=1e-9)
        assert lat_min == pytest.approx(50.064 + 32 * (-0.001), abs=1e-9)


# ---------------------------------------------------------------------------
# build_manifest
# ---------------------------------------------------------------------------

class TestBuildManifest:
    """Tests for build_manifest."""

    def test_length_equals_total_chips(self):
        """Returned list has one entry per chip in the grid."""
        rows = build_manifest(_GRID_GEO)
        assert len(rows) == _GRID_GEO.total

    def test_no_chip_stats_defaults(self):
        """When chip_stats=None, cloud_pct, variance, and rejected get safe defaults."""
        rows = build_manifest(_GRID_GEO, chip_stats=None)
        for row in rows:
            assert row["cloud_pct"] == "0.0000"
            assert row["variance"] == "0.00"
            assert row["rejected"] == "false"

    def test_chip_stats_values_appear_in_correct_row(self):
        """Stats for a specific chip_index land in the matching manifest row."""
        stats = [{"chip_index": 2, "cloud_pct": 12.5, "variance": 300.0, "rejected": True}]
        rows = build_manifest(_GRID_GEO, chip_stats=stats)
        # Row at list position 2 corresponds to chip_index 2
        row2 = rows[2]
        assert row2["chip_index"] == 2
        assert row2["cloud_pct"] == "12.5000"
        assert row2["variance"] == "300.00"
        assert row2["rejected"] == "true"
        # Other rows stay at defaults
        assert rows[0]["cloud_pct"] == "0.0000"

    def test_rowcol_naming_pattern(self):
        """naming='rowcol' produces chip_r{row:04d}_c{col:04d}.png filenames."""
        rows = build_manifest(_GRID_GEO, naming="rowcol", fmt_ext=".png")
        for row in rows:
            expected = f"chip_r{row['row']:04d}_c{row['col']:04d}.png"
            assert row["filename"] == expected

    @pytest.mark.parametrize("idx,expected_row,expected_col", [
        (0, 0, 0),
        (1, 0, 1),
        (2, 1, 0),
        (3, 1, 1),
    ])
    def test_row_col_indices_match_flat_index(self, idx, expected_row, expected_col):
        """Row and col fields match the flat index divided by n_cols."""
        rows = build_manifest(_GRID_GEO)
        assert rows[idx]["row"] == expected_row
        assert rows[idx]["col"] == expected_col

    @pytest.mark.parametrize("idx,col_off,chip_w", [
        (0, 0, 32),
        (1, 32, 32),
        (2, 0, 32),
        (3, 32, 32),
    ])
    def test_pixel_bounding_box(self, idx, col_off, chip_w):
        """pixel_x_min == col_off and pixel_x_max == col_off + chip_w."""
        rows = build_manifest(_GRID_GEO)
        row = rows[idx]
        assert row["pixel_x_min"] == col_off
        assert row["pixel_x_max"] == col_off + chip_w

    def test_no_crs_gives_empty_geo_fields(self):
        """When the grid has no CRS, lon/lat fields are empty strings."""
        rows = build_manifest(_GRID_NOCRS)
        for row in rows:
            assert row["lon_min"] == ""
            assert row["lat_min"] == ""
            assert row["lon_max"] == ""
            assert row["lat_max"] == ""

    def test_geo_fields_populated_for_geographic_crs(self):
        """With a geographic CRS every chip has non-empty lon/lat fields."""
        rows = build_manifest(_GRID_GEO)
        for row in rows:
            assert row["lon_min"] != ""
            assert row["lat_min"] != ""

    def test_chip_index_field_matches_position(self):
        """chip_index in each row equals its position in the list."""
        rows = build_manifest(_GRID_GEO)
        for i, row in enumerate(rows):
            assert row["chip_index"] == i


# ---------------------------------------------------------------------------
# write_manifest_csv
# ---------------------------------------------------------------------------

class TestWriteManifestCsv:
    """Tests for write_manifest_csv."""

    def test_empty_list_returns_empty_bytes(self):
        """An empty rows list produces b''."""
        assert write_manifest_csv([]) == b""

    def test_output_is_valid_utf8(self):
        """The output can be decoded as UTF-8 without error."""
        rows = build_manifest(_GRID_GEO)
        csv_bytes = write_manifest_csv(rows)
        csv_bytes.decode("utf-8")  # raises if not valid UTF-8

    def test_single_row_has_header_and_data_row(self):
        """A single-chip manifest produces exactly 2 non-empty lines: header + data."""
        # Build a 1-chip grid using a 32×32 image with a 32×32 chip size
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        transform = from_bounds(10.0, 50.0, 10.032, 50.032, 32, 32)
        meta = {"crs": CRS.from_epsg(4326), "transform": transform,
                "dtype": "uint8", "count": 3, "driver": "GTiff"}
        grid = build_chip_grid(img, meta, chip_w=32, chip_h=32)

        rows = build_manifest(grid)
        csv_bytes = write_manifest_csv(rows)
        lines = [l for l in csv_bytes.decode("utf-8").splitlines() if l]
        assert len(lines) == 2

    def test_round_trip_preserves_values(self):
        """Parsing the CSV back yields the same values as the original row dict."""
        rows = build_manifest(_GRID_GEO)
        csv_bytes = write_manifest_csv(rows)

        reader = csv.DictReader(io.StringIO(csv_bytes.decode("utf-8")))
        parsed = list(reader)

        first_in = rows[0]
        first_out = parsed[0]

        for key in first_in:
            assert first_out[key] == str(first_in[key]), (
                f"Mismatch on key '{key}': {first_out[key]!r} != {first_in[key]!r}"
            )

    def test_header_contains_expected_keys(self):
        """The CSV header includes all expected manifest column names."""
        expected_keys = {
            "chip_index", "row", "col",
            "pixel_x_min", "pixel_y_min", "pixel_x_max", "pixel_y_max",
            "lon_min", "lat_min", "lon_max", "lat_max",
            "cloud_pct", "variance", "filename", "rejected",
        }
        rows = build_manifest(_GRID_GEO)
        csv_bytes = write_manifest_csv(rows)
        reader = csv.DictReader(io.StringIO(csv_bytes.decode("utf-8")))
        assert expected_keys.issubset(set(reader.fieldnames))
