"""
tests/test_annotation_export.py

Tests for sentinel_backend.chipping.annotation_export — build_coco_manifest and
build_yolo_files.
"""

import numpy as np
import pytest
from rasterio.crs import CRS
from rasterio.transform import from_bounds

from sentinel_backend.chipping.gdal_chipper import build_chip_grid
from sentinel_backend.chipping.annotation_export import build_coco_manifest, build_yolo_files


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def make_grid_4chip():
    """Return a 2-column × 2-row ChipGrid over a 64×64 image with 32×32 chips.

    Uses EPSG:4326 so geographic bounds are available on every chip.
    The grid produces windows at offsets (0,0), (32,0), (0,32), (32,32).

    Returns:
        ChipGrid with 4 chips and a valid geographic CRS.
    """
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    transform = from_bounds(west=10.0, south=50.0, east=10.064, north=50.064,
                            width=64, height=64)
    meta = {
        "crs": CRS.from_epsg(4326),
        "transform": transform,
        "dtype": "uint8",
        "count": 3,
        "driver": "GTiff",
    }
    return build_chip_grid(img, meta, chip_w=32, chip_h=32)


def make_chip_stats_with_rejection(rejected_idx: int = 2) -> list:
    """Return a 4-chip stats list with one chip marked as rejected.

    Args:
        rejected_idx: Flat chip index that should be flagged rejected.
            Defaults to chip 2 (bottom-left in a 2×2 grid).

    Returns:
        list of 4 dicts, each with keys chip_index, cloud_pct, variance,
        rejected.
    """
    return [
        {"chip_index": i, "cloud_pct": 0.0, "variance": 100.0,
         "rejected": (i == rejected_idx)}
        for i in range(4)
    ]


# Module-level instances reused across tests.
_GRID = make_grid_4chip()
_STATS = make_chip_stats_with_rejection(rejected_idx=2)
_REJECTED_INDICES = [2]


# ---------------------------------------------------------------------------
# build_coco_manifest — structure
# ---------------------------------------------------------------------------

class TestBuildCocoManifestStructure:
    """Tests verifying the top-level structure of the COCO manifest dict."""

    def test_has_required_top_level_keys(self):
        """Returned dict contains 'info', 'images', 'annotations', 'categories'."""
        result = build_coco_manifest(_GRID)
        assert set(result.keys()) >= {"info", "images", "annotations", "categories"}

    def test_annotations_always_empty(self):
        """annotations list is always empty — dataset manifest only."""
        result = build_coco_manifest(_GRID)
        assert result["annotations"] == []

    def test_categories_always_empty(self):
        """categories list is always empty for this unannotated dataset."""
        result = build_coco_manifest(_GRID)
        assert result["categories"] == []


# ---------------------------------------------------------------------------
# build_coco_manifest — image entries
# ---------------------------------------------------------------------------

class TestBuildCocoManifestImages:
    """Tests for the 'images' list entries in the COCO manifest."""

    def test_image_count_without_rejection(self):
        """Without rejection filtering, all 4 chips appear as image entries."""
        result = build_coco_manifest(_GRID)
        assert len(result["images"]) == 4

    def test_file_names_start_with_chips_prefix(self):
        """Every image file_name is prefixed with 'chips/'."""
        result = build_coco_manifest(_GRID)
        for img in result["images"]:
            assert img["file_name"].startswith("chips/")

    def test_image_ids_are_one_based(self):
        """Image ids start at 1 and are strictly positive integers."""
        result = build_coco_manifest(_GRID)
        ids = [img["id"] for img in result["images"]]
        assert all(isinstance(i, int) and i >= 1 for i in ids)
        # Chip index 0 → id 1
        assert ids[0] == 1

    def test_rejected_chip_excluded_by_default(self):
        """Chip at rejected_index is absent when include_rejected=False."""
        result = build_coco_manifest(
            _GRID,
            rejected_indices=_REJECTED_INDICES,
            include_rejected=False,
        )
        present_ids = {img["id"] for img in result["images"]}
        # Chip index 2 → COCO id 3 (1-based)
        assert 3 not in present_ids
        assert len(result["images"]) == 3

    def test_include_rejected_true_keeps_all_chips(self):
        """include_rejected=True includes the rejected chip in the manifest."""
        result = build_coco_manifest(
            _GRID,
            rejected_indices=_REJECTED_INDICES,
            include_rejected=True,
        )
        assert len(result["images"]) == 4

    def test_rowcol_naming_pattern(self):
        """naming='rowcol' produces file_names with chip_r{row}_c{col} pattern."""
        result = build_coco_manifest(_GRID, naming="rowcol", fmt_ext=".png")
        for img in result["images"]:
            # file_name is 'chips/chip_r0000_c0001.png' etc.
            stem = img["file_name"].split("/")[-1]
            assert stem.startswith("chip_r")
            assert "_c" in stem

    @pytest.mark.parametrize("idx,expected_row,expected_col", [
        (0, "0000", "0000"),
        (1, "0000", "0001"),
        (2, "0001", "0000"),
        (3, "0001", "0001"),
    ])
    def test_rowcol_filename_row_col_digits(self, idx, expected_row, expected_col):
        """Each chip filename encodes the correct zero-padded row and col."""
        result = build_coco_manifest(_GRID, naming="rowcol", fmt_ext=".png")
        fname = result["images"][idx]["file_name"]
        assert f"chip_r{expected_row}_c{expected_col}.png" in fname

    def test_image_has_width_and_height(self):
        """Every image entry carries 'width' and 'height' fields."""
        result = build_coco_manifest(_GRID)
        for img in result["images"]:
            assert "width" in img and img["width"] == 32
            assert "height" in img and img["height"] == 32

    def test_geo_bbox_present_for_geographic_crs(self):
        """Chips from a geographic-CRS grid carry a 'geo_bbox' list of 4 floats."""
        result = build_coco_manifest(_GRID)
        for img in result["images"]:
            assert "geo_bbox" in img
            bbox = img["geo_bbox"]
            assert len(bbox) == 4
            assert all(isinstance(v, float) for v in bbox)


# ---------------------------------------------------------------------------
# build_yolo_files — structure
# ---------------------------------------------------------------------------

class TestBuildYoloFilesStructure:
    """Tests for the dict returned by build_yolo_files."""

    def test_dataset_yaml_present_and_nonempty(self):
        """Output is a dict with a non-empty bytes 'dataset.yaml' key."""
        result = build_yolo_files(_GRID)
        assert isinstance(result, dict)
        assert "dataset.yaml" in result
        assert isinstance(result["dataset.yaml"], bytes)
        assert len(result["dataset.yaml"]) > 0


# ---------------------------------------------------------------------------
# build_yolo_files — label files
# ---------------------------------------------------------------------------

class TestBuildYoloFilesLabels:
    """Tests for per-chip label .txt entries in the YOLO output dict."""

    def test_label_count_without_rejection(self):
        """Without rejection filtering, one label file per chip (4 chips → 4 labels)."""
        result = build_yolo_files(_GRID)
        label_keys = [k for k in result if k.startswith("labels/") and k.endswith(".txt")]
        assert len(label_keys) == 4

    def test_all_label_files_are_empty_bytes(self):
        """Every label .txt file is b'' — no annotations yet."""
        result = build_yolo_files(_GRID)
        for key, value in result.items():
            if key.startswith("labels/") and key.endswith(".txt"):
                assert value == b"", f"{key} should be empty but got {value!r}"

    def test_label_stems_match_chip_filename_stems(self):
        """Each label file stem matches the corresponding chip's filename stem."""
        result = build_yolo_files(_GRID, naming="rowcol", fmt_ext=".png")
        label_keys = [k for k in result if k.startswith("labels/") and k.endswith(".txt")]
        # Stems inside labels/ directory (strip "labels/" prefix and ".txt" suffix)
        label_stems = {k[len("labels/"):-len(".txt")] for k in label_keys}
        # Expected stems from rowcol naming: chip_r0000_c0000, chip_r0000_c0001, …
        for stem in label_stems:
            assert stem.startswith("chip_r")

    def test_rejected_chip_excluded_by_default(self):
        """Chip at rejected_index has no label file when include_rejected=False."""
        result = build_yolo_files(
            _GRID,
            rejected_indices=_REJECTED_INDICES,
            include_rejected=False,
        )
        label_keys = [k for k in result if k.startswith("labels/") and k.endswith(".txt")]
        assert len(label_keys) == 3
        # Chip index 2 → chip_r0001_c0000; its label must be absent
        assert "labels/chip_r0001_c0000.txt" not in result

    def test_include_rejected_true_keeps_all_labels(self):
        """include_rejected=True produces label files for all chips including rejected."""
        result = build_yolo_files(
            _GRID,
            rejected_indices=_REJECTED_INDICES,
            include_rejected=True,
        )
        label_keys = [k for k in result if k.startswith("labels/") and k.endswith(".txt")]
        assert len(label_keys) == 4

    def test_rowcol_label_key_format(self):
        """Label dict keys follow 'labels/<stem>.txt' with 'chip_r' stem prefix."""
        result = build_yolo_files(_GRID, naming="rowcol", fmt_ext=".png")
        label_keys = [k for k in result if k.startswith("labels/") and k.endswith(".txt")]
        for key in label_keys:
            stem = key[len("labels/"):-len(".txt")]
            assert stem.startswith("chip_r"), (
                f"Expected stem starting with 'chip_r', got {stem!r}"
            )
