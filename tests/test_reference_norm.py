"""Tests for processing/reference_norm.py"""
import pickle
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))

from processing.reference_norm import compute_reference_stats, apply_reference_normalisation


def _make_image(mean_rgb, std=20, size=(64, 64)):
    """Create a reproducible random uint8 (H, W, 3) image."""
    rng = np.random.default_rng(42)
    img = np.zeros((size[0], size[1], 3), dtype=np.float32)
    for c in range(3):
        img[:, :, c] = rng.normal(mean_rgb[c], std, size)
    return np.clip(img, 0, 255).astype(np.uint8)


# Module-level stats shared across single-image tests.
_SINGLE_IMG = _make_image([100, 120, 80])
_SINGLE_STATS = compute_reference_stats([_SINGLE_IMG])


# --- compute_reference_stats ---

def test_compute_single_image_structure():
    """Stats dict has correct n, shape/dtype for mean/std, and valid CDFs."""
    stats = _SINGLE_STATS
    assert stats["n"] == 1
    assert stats["mean"].shape == (3,) and stats["mean"].dtype == np.float32
    assert stats["std"].shape == (3,) and stats["std"].dtype == np.float32
    assert len(stats["cdfs"]) == 3
    for cdf in stats["cdfs"]:
        assert cdf.shape == (256,) and cdf.dtype == np.float32
        assert cdf.min() >= 0.0 and cdf.max() <= 1.0 + 1e-5

def test_compute_multiple_images_averaging():
    img_a = _make_image([80, 80, 80])
    img_b = _make_image([160, 160, 160])
    stats_a = compute_reference_stats([img_a])
    stats_b = compute_reference_stats([img_b])
    stats_ab = compute_reference_stats([img_a, img_b])
    assert stats_ab["n"] == 2
    # Averaged mean should be between individual means
    for c in range(3):
        lo = min(stats_a["mean"][c], stats_b["mean"][c])
        hi = max(stats_a["mean"][c], stats_b["mean"][c])
        assert lo <= stats_ab["mean"][c] <= hi

def test_compute_empty_raises():
    with pytest.raises(ValueError):
        compute_reference_stats([])

def test_cdfs_monotonic():
    for cdf in _SINGLE_STATS["cdfs"]:
        assert np.all(np.diff(cdf) >= -1e-6), "CDF must be non-decreasing"

def test_picklable():
    data = pickle.dumps(_SINGLE_STATS)
    restored = pickle.loads(data)
    assert restored["n"] == _SINGLE_STATS["n"]
    np.testing.assert_array_equal(restored["mean"], _SINGLE_STATS["mean"])


# --- apply_reference_normalisation ---

def _ref_stats(mean_rgb):
    img = _make_image(mean_rgb)
    return compute_reference_stats([img])

@pytest.mark.parametrize("method", ["histogram", "linear"])
def test_apply_output_properties(method):
    """Both normalisation methods return uint8, preserve shape, and stay in [0, 255]."""
    img = _make_image([100, 100, 100])
    out = apply_reference_normalisation(img, _ref_stats([150, 150, 150]), method)
    assert out.dtype == np.uint8
    assert out.shape == img.shape
    assert out.min() >= 0
    assert out.max() <= 255

def test_apply_histogram_shifts_mean():
    img = _make_image([80, 80, 80])
    ref_stats = _ref_stats([180, 180, 180])
    out = apply_reference_normalisation(img, ref_stats, "histogram")
    # Output mean should be closer to 180 than the original 80
    assert abs(out.mean() - 180) < abs(img.mean() - 180)

def test_apply_linear_shifts_statistics():
    img = _make_image([80, 80, 80])
    ref_stats = _ref_stats([160, 160, 160])
    out = apply_reference_normalisation(img, ref_stats, "linear")
    # Each channel mean should be closer to 160 than original 80
    for c in range(3):
        orig_dist = abs(img[:,:,c].mean() - 160)
        out_dist  = abs(out[:,:,c].mean()  - 160)
        assert out_dist < orig_dist

def test_apply_invalid_method_raises():
    img = _make_image([100, 100, 100])
    with pytest.raises(ValueError):
        apply_reference_normalisation(img, _ref_stats([150, 150, 150]), "bogus")

def test_apply_linear_degenerate_reference_band():
    """Channel with near-zero reference std should be returned unchanged."""
    img = _make_image([100, 100, 100])
    stats = _ref_stats([150, 150, 150])
    # Force one channel's std to zero
    stats["std"][0] = 0.0
    out = apply_reference_normalisation(img, stats, "linear")
    np.testing.assert_array_equal(out[:, :, 0], img[:, :, 0])
