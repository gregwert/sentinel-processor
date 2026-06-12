"""
tests/test_chip_filter.py

Tests for app.chipping.chip_filter — compute_chip_stats and apply_chip_filters.
"""

import sys
from unittest.mock import patch

import numpy as np
import pytest
from rasterio.transform import from_bounds

from app.chipping.chip_filter import apply_chip_filters, compute_chip_stats
import app.chipping.gdal_chipper as _gdal_chipper_mod
from app.chipping.gdal_chipper import build_chip_grid, get_chip


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _make_meta():
    """Return a minimal source_meta dict containing only a transform.

    No CRS is included; filter tests never export chips to disk so the
    transform is sufficient for chip_affine to run.  The transform maps a
    64x64 pixel image onto a unit-square bounding box.
    """
    return {"transform": from_bounds(0, 0, 1, 1, 64, 64)}


def _make_grid(img: np.ndarray):
    """Build a 2x2 ChipGrid (32x32 chips, no overlap) from a 64x64 image.

    Args:
        img: uint8 array of shape (64, 64, 3).

    Returns:
        ChipGrid with 4 chips, each nominally 32x32 pixels.
    """
    return build_chip_grid(img, _make_meta(), chip_w=32, chip_h=32, overlap=0.0)


def _solid_img(value: int) -> np.ndarray:
    """Return a 64x64x3 uint8 image filled with a single constant pixel value.

    Args:
        value: Pixel fill value in [0, 255].
    """
    return np.full((64, 64, 3), value, dtype=np.uint8)


def _varied_img() -> np.ndarray:
    """Return a 64x64x3 uint8 image with high pixel-to-pixel variance."""
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)


# Reusable chip arrays for compute_chip_stats unit tests.
_CHIP_FLAT = np.zeros((32, 32, 3), dtype=np.uint8)
_CHIP_HIGH_VAR = np.arange(32 * 32 * 3, dtype=np.uint8).reshape(32, 32, 3)


# ---------------------------------------------------------------------------
# apply_chip_filters: import-path shim
#
# chip_filter.py does `from chipping.gdal_chipper import get_chip` inside the
# function body.  The app is launched with app/ on sys.path so that bare
# import resolves to the chipping sub-package directly.  From the test
# runner (project root), app.chipping.gdal_chipper is already imported.
# We register the same module object under the bare key so the lazy import
# inside apply_chip_filters always finds the real get_chip.
# ---------------------------------------------------------------------------

def _ensure_bare_chipping_alias():
    """Register app.chipping.gdal_chipper under 'chipping.gdal_chipper' in sys.modules.

    This lets `from chipping.gdal_chipper import get_chip` inside
    apply_chip_filters resolve to the already-imported module rather than
    raising ModuleNotFoundError.
    """
    import app.chipping as _pkg
    sys.modules.setdefault("chipping", _pkg)
    sys.modules.setdefault("chipping.gdal_chipper", _gdal_chipper_mod)


_ensure_bare_chipping_alias()


# ---------------------------------------------------------------------------
# compute_chip_stats
# ---------------------------------------------------------------------------

def test_compute_chip_stats_no_cloud_mask():
    """cloud_pct is 0.0 when cloud_mask is None."""
    stats = compute_chip_stats(_CHIP_FLAT, None, col_off=0, row_off=0)
    assert stats["cloud_pct"] == 0.0


def test_compute_chip_stats_full_cloud_mask():
    """cloud_pct is 1.0 when the entire mask patch covering the chip is True."""
    mask = np.ones((64, 64), dtype=bool)
    stats = compute_chip_stats(_CHIP_FLAT, mask, col_off=0, row_off=0)
    assert stats["cloud_pct"] == pytest.approx(1.0)


def test_compute_chip_stats_half_cloud_mask():
    """cloud_pct is 0.5 when exactly half the mask patch is True."""
    mask = np.zeros((64, 64), dtype=bool)
    mask[0:16, 0:32] = True  # top 16 of the 32-row patch
    stats = compute_chip_stats(_CHIP_FLAT, mask, col_off=0, row_off=0)
    assert stats["cloud_pct"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    "chip, expect_high",
    [
        (_CHIP_HIGH_VAR, True),
        (_CHIP_FLAT, False),
    ],
    ids=["high_variance", "zero_variance"],
)
def test_compute_chip_stats_variance(chip, expect_high):
    """variance reflects actual pixel spread: non-zero for varied chips, zero for flat."""
    stats = compute_chip_stats(chip, None, col_off=0, row_off=0)
    if expect_high:
        assert stats["variance"] > 0
    else:
        assert stats["variance"] == pytest.approx(0.0)


def test_compute_chip_stats_edge_chip_uses_slice_size_as_denominator():
    """Edge chip: cloud_pct denominator equals the actual chip pixel count.

    A chip_array of shape (16, 16, 3) at offset (48, 48) slices a 16x16
    patch from the mask.  Marking all 256 pixels as cloud must yield
    cloud_pct == 1.0, confirming the real slice dimensions drive the
    denominator rather than any nominal (larger) chip size.
    """
    edge_chip = np.zeros((16, 16, 3), dtype=np.uint8)
    mask = np.zeros((64, 64), dtype=bool)
    mask[48:64, 48:64] = True  # exactly the 16x16 region at the edge
    stats = compute_chip_stats(edge_chip, mask, col_off=48, row_off=48)
    assert stats["cloud_pct"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# apply_chip_filters
# ---------------------------------------------------------------------------

def test_apply_chip_filters_both_disabled_accepts_all():
    """All chips are accepted when both cloud and variance filters are disabled."""
    grid = _make_grid(_varied_img())
    accepted, rejected, _ = apply_chip_filters(
        grid, None,
        enable_cloud_filter=False,
        enable_variance_filter=False,
    )
    assert sorted(accepted) == list(range(grid.total))
    assert rejected == []


def test_apply_chip_filters_cloud_only_rejects_above_threshold():
    """Cloud filter rejects chips whose cloud_pct exceeds the threshold.

    The 64x64 image has a constant pixel value so variance is uniform.
    The bottom half (row >= 32) is masked as cloud: chips 2 and 3.
    With cloud_thresh=0.5 those chips are rejected; chips 0 and 1 are accepted.
    """
    grid = _make_grid(_solid_img(128))
    mask = np.zeros((64, 64), dtype=bool)
    mask[32:, :] = True  # fully clouds chips 2 and 3 (bottom row)

    accepted, rejected, _ = apply_chip_filters(
        grid, mask,
        cloud_thresh=0.5,
        enable_cloud_filter=True,
        enable_variance_filter=False,
    )
    assert sorted(accepted) == [0, 1]
    assert sorted(rejected) == [2, 3]


def test_apply_chip_filters_variance_only_rejects_below_threshold():
    """Variance filter rejects flat chips and accepts high-variance chips.

    Top row of the image (chips 0 and 1) is all zeros — variance = 0.
    Bottom row (chips 2 and 3) has a column ramp — variance is high.
    With variance_thresh=500 only the bottom chips pass.
    """
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[32:, :, :] = np.linspace(0, 255, 32, dtype=np.uint8)[:, None, None]
    grid = _make_grid(img)

    accepted, rejected, _ = apply_chip_filters(
        grid, None,
        variance_thresh=500,
        enable_cloud_filter=False,
        enable_variance_filter=True,
    )
    assert sorted(rejected) == [0, 1]
    assert sorted(accepted) == [2, 3]


def test_apply_chip_filters_both_enabled_union_rejection():
    """A chip failing either filter is rejected; only chips passing both are accepted.

    chips 0,1: flat (fail variance); chip 2: fully clouded (fails cloud); chip 3: passes.
    """
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[32:, :, :] = np.linspace(0, 255, 32, dtype=np.uint8)[:, None, None]
    mask = np.zeros((64, 64), dtype=bool)
    mask[32:, :32] = True  # clouds chip 2 only

    grid = _make_grid(img)
    accepted, rejected, _ = apply_chip_filters(
        grid, mask,
        cloud_thresh=0.5,
        variance_thresh=500,
        enable_cloud_filter=True,
        enable_variance_filter=True,
    )
    assert accepted == [3]
    assert sorted(rejected) == [0, 1, 2]


def test_apply_chip_filters_accepted_rejected_partition():
    """accepted and rejected together partition the full chip index set with no overlap."""
    grid = _make_grid(_varied_img())
    mask = np.zeros((64, 64), dtype=bool)
    mask[32:, :] = True

    accepted, rejected, _ = apply_chip_filters(
        grid, mask,
        cloud_thresh=0.3,
        variance_thresh=100,
        enable_cloud_filter=True,
        enable_variance_filter=True,
    )
    combined = sorted(accepted + rejected)
    assert combined == list(range(grid.total))
    assert set(accepted).isdisjoint(set(rejected))


def test_apply_chip_filters_chip_stats_covers_every_index():
    """chip_stats has one entry per chip and each entry carries its chip_index."""
    grid = _make_grid(_varied_img())
    _, _, stats = apply_chip_filters(
        grid, None,
        enable_cloud_filter=False,
        enable_variance_filter=False,
    )
    assert len(stats) == grid.total
    assert [s["chip_index"] for s in stats] == list(range(grid.total))
