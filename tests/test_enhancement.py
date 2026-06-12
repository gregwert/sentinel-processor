"""
tests/test_enhancement.py

Tests for app.processing.enhancement — apply_clahe and apply_gray_world.
"""

import numpy as np

from app.processing.enhancement import apply_clahe, apply_gray_world


def make_low_contrast_image(h=128, w=128):
    """Image with values clustered near 128 — low contrast."""
    rng = np.random.default_rng(42)
    img = (rng.normal(128, 5, (h, w, 3))).clip(0, 255).astype(np.uint8)
    return img


# Module-level images shared across sanity tests.
_IMG_CLAHE = make_low_contrast_image()
_IMG_GW = make_low_contrast_image()


# ---------------------------------------------------------------------------
# apply_clahe tests
# ---------------------------------------------------------------------------

def test_clahe_output_properties():
    """apply_clahe returns uint8 with correct shape, bounded values."""
    result = apply_clahe(_IMG_CLAHE)
    assert result.dtype == np.uint8
    assert result.shape == _IMG_CLAHE.shape
    assert int(result.min()) >= 0
    assert int(result.max()) <= 255


def test_clahe_increases_contrast():
    """CLAHE should increase the standard deviation of a low-contrast image."""
    result = apply_clahe(_IMG_CLAHE)
    assert result.astype(np.float32).std() > _IMG_CLAHE.astype(np.float32).std()


# ---------------------------------------------------------------------------
# apply_gray_world tests
# ---------------------------------------------------------------------------

def test_gray_world_output_properties():
    """apply_gray_world returns uint8 with correct shape, bounded values."""
    result = apply_gray_world(_IMG_GW)
    assert result.dtype == np.uint8
    assert result.shape == _IMG_GW.shape
    assert int(result.min()) >= 0
    assert int(result.max()) <= 255


def test_gray_world_balances_channels():
    """Gray World should reduce the spread of per-channel means."""
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    img[:, :, 0] = (img[:, :, 0] * 0.3).astype(np.uint8)  # heavily biased red
    result = apply_gray_world(img)
    in_means = img.astype(np.float32).mean(axis=(0, 1))
    out_means = result.astype(np.float32).mean(axis=(0, 1))
    assert out_means.std() < in_means.std()
