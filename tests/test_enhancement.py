"""
tests/test_enhancement.py

Tests for app.processing.enhancement — apply_clahe and apply_standardization.
"""

import numpy as np
import pytest

from app.processing.enhancement import apply_clahe, apply_standardization


def make_low_contrast_image(h=128, w=128):
    """Image with values clustered near 128 — low contrast."""
    rng = np.random.default_rng(42)
    img = (rng.normal(128, 5, (h, w, 3))).clip(0, 255).astype(np.uint8)
    return img


# ---------------------------------------------------------------------------
# apply_clahe tests
# ---------------------------------------------------------------------------

def test_clahe_output_dtype():
    img = make_low_contrast_image()
    result = apply_clahe(img)
    assert result.dtype == np.uint8


def test_clahe_output_shape():
    img = make_low_contrast_image(64, 96)
    result = apply_clahe(img)
    assert result.shape == img.shape


def test_clahe_output_range():
    img = make_low_contrast_image()
    result = apply_clahe(img)
    assert int(result.min()) >= 0
    assert int(result.max()) <= 255


def test_clahe_increases_contrast():
    """CLAHE should increase the standard deviation of a low-contrast image."""
    img = make_low_contrast_image()
    result = apply_clahe(img)
    assert result.astype(np.float32).std() > img.astype(np.float32).std()


# ---------------------------------------------------------------------------
# apply_standardization tests
# ---------------------------------------------------------------------------

def test_standardization_output_dtype():
    img = make_low_contrast_image()
    result = apply_standardization(img)
    assert result.dtype == np.uint8


def test_standardization_output_range():
    img = make_low_contrast_image()
    result = apply_standardization(img)
    assert int(result.min()) >= 0
    assert int(result.max()) <= 255


def test_standardization_mean_approx():
    """Output mean should be within 20 of the target mean (127.5)."""
    img = make_low_contrast_image()
    result = apply_standardization(img, target_mean=127.5, target_std=45.0)
    assert abs(result.astype(np.float32).mean() - 127.5) < 20
