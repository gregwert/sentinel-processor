"""
tests/test_dehazing.py

Tests for app.processing.dehazing — dark channel prior pipeline functions.
"""

import numpy as np
import pytest

from app.processing.dehazing import (
    detect_clouds_simple,
    dark_channel,
    estimate_atmospheric_light,
    transmission_map,
    guided_filter_transmission,
    recover_scene_radiance,
    dehaze,
)


def make_hazy_image(h=128, w=128, value=180):
    """Synthetic hazy-looking uint8 RGB image — flat uniform gray."""
    return np.full((h, w, 3), value, dtype=np.uint8)


# ---------------------------------------------------------------------------
# dehaze() end-to-end tests
# ---------------------------------------------------------------------------

def test_dehaze_output_dtype():
    img = make_hazy_image()
    result = dehaze(img)
    assert result.dtype == np.uint8


def test_dehaze_output_shape():
    img = make_hazy_image(64, 96)
    result = dehaze(img)
    assert result.shape == img.shape


def test_dehaze_no_nan():
    img = make_hazy_image()
    result = dehaze(img)
    assert not np.isnan(result.astype(np.float32)).any()


def test_dehaze_values_in_range():
    img = make_hazy_image()
    result = dehaze(img)
    assert int(result.min()) >= 0
    assert int(result.max()) <= 255


def test_dehaze_brightens_hazy_image():
    """DCP should increase contrast (std) on a hazy / washed-out image.

    Atmospheric haze compresses dynamic range: the darkest pixels are pushed up
    by the haze veil and the brightest are pulled down by scattering.  DCP
    inverts this by estimating and removing the atmospheric component, which
    increases the standard deviation (spread) of pixel values.  We construct a
    scene with low saturation and moderate brightness (simulating a hazy
    wash-out) and assert that the output has higher std than the input.
    A perfectly flat image would leave std unchanged; adding a small amount of
    structure ensures the dark channel is non-zero so the filter actually fires.
    """
    rng = np.random.default_rng(42)
    # Clear-scene objects: coloured objects with moderate dynamic range
    objects = rng.integers(20, 100, size=(128, 128, 3), dtype=np.uint8)
    # Atmospheric haze: additive gray veil that washes everything out
    haze_veil = 130
    img = np.clip(objects.astype(np.int32) + haze_veil, 0, 255).astype(np.uint8)
    result = dehaze(img, mask_clouds=False)
    # Dehazing removes the veil → the clear-scene objects become more saturated
    # and the pixel value spread (std) must increase.
    assert result.astype(np.float32).std() > img.astype(np.float32).std()


# ---------------------------------------------------------------------------
# detect_clouds_simple tests
# ---------------------------------------------------------------------------

def test_cloud_detection_bright_white():
    """Mostly-white image should be detected as cloud (>50% True)."""
    img = np.full((64, 64, 3), 240, dtype=np.uint8)
    mask = detect_clouds_simple(img)
    assert mask.mean() > 0.5


def test_cloud_detection_dark_vegetated():
    """Dark greenish image should have 0% cloud pixels."""
    img = np.full((64, 64, 3), [30, 80, 30], dtype=np.uint8)
    mask = detect_clouds_simple(img)
    assert mask.sum() == 0


# ---------------------------------------------------------------------------
# Cloud pixel restoration test
# ---------------------------------------------------------------------------

def test_cloud_pixels_restored():
    """Top half is white (cloud); bottom half is colored land. Cloud pixels
    in the output should be identical to the input after mask_clouds=True."""
    h, w = 64, 64
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # Top half: white cloud region
    img[:h // 2, :, :] = 240
    # Bottom half: dark greenish land
    img[h // 2:, :, :] = [30, 80, 30]

    result = dehaze(img, mask_clouds=True)
    top_in = img[:h // 2, :, :]
    top_out = result[:h // 2, :, :]
    np.testing.assert_array_equal(top_out, top_in)


# ---------------------------------------------------------------------------
# estimate_atmospheric_light fallback test
# ---------------------------------------------------------------------------

def test_atmospheric_light_excludes_clouds():
    """When >80% of the scene is cloud, the function should fall back to all
    pixels and return a valid float32 (3,) array without raising."""
    h, w = 64, 64
    img = np.full((h, w, 3), 240, dtype=np.uint8)
    img_f32 = img.astype(np.float32) / 255.0

    # Build dark channel manually
    dark = dark_channel(img_f32)

    # Cloud mask that marks almost the entire image as cloud
    cloud_mask = np.ones((h, w), dtype=bool)
    cloud_mask[60:, 60:] = False  # tiny non-cloud region

    A = estimate_atmospheric_light(img_f32, dark, cloud_mask=cloud_mask)

    assert A is not None
    assert A.dtype == np.float32
    assert A.shape == (3,)
    assert not np.isnan(A).any()


# ---------------------------------------------------------------------------
# dark_channel shape test
# ---------------------------------------------------------------------------

def test_dark_channel_shape():
    h, w = 50, 70
    img_f32 = np.random.rand(h, w, 3).astype(np.float32)
    dc = dark_channel(img_f32)
    assert dc.shape == (h, w)
