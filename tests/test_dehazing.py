"""
tests/test_dehazing.py

Tests for app.processing.dehazing — dark channel prior pipeline functions.
"""

import numpy as np

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

def test_dehaze_output_properties():
    """Output has same shape as input, correct dtype, bounded values, and no NaN.

    Uses a non-square image (64×96) to confirm H and W are not swapped.
    """
    img = make_hazy_image(64, 96)
    result = dehaze(img)
    assert result.dtype == np.uint8
    assert result.shape == img.shape
    assert not np.isnan(result.astype(np.float32)).any()
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


# ---------------------------------------------------------------------------
# transmission_map tests
# ---------------------------------------------------------------------------

def test_transmission_map_shape_and_dtype():
    """transmission_map returns float32 with shape (H, W)."""
    h, w = 64, 64
    img_f32 = np.random.rand(h, w, 3).astype(np.float32)
    A = np.array([0.9, 0.85, 0.88], dtype=np.float32)
    t = transmission_map(img_f32, A)
    assert t.shape == (h, w)
    assert t.dtype == np.float32


def test_transmission_map_values_in_range():
    """Transmission values must lie in (0.0, 1.0] — no zeros or values above 1."""
    rng = np.random.default_rng(7)
    img_f32 = rng.random((64, 64, 3)).astype(np.float32)
    A = np.array([0.8, 0.8, 0.8], dtype=np.float32)
    t = transmission_map(img_f32, A, patch_size=5, omega=0.95)
    assert float(t.min()) > 0.0
    assert float(t.max()) <= 1.0 + 1e-5  # small tolerance for float arithmetic


# ---------------------------------------------------------------------------
# recover_scene_radiance tests
# ---------------------------------------------------------------------------

def test_recover_scene_radiance_shape_and_dtype():
    """Output shape and dtype must match the input image."""
    h, w = 48, 64
    img_f32 = np.random.rand(h, w, 3).astype(np.float32)
    t = np.full((h, w), 0.5, dtype=np.float32)
    A = np.array([0.9, 0.85, 0.88], dtype=np.float32)
    J = recover_scene_radiance(img_f32, t, A)
    assert J.shape == img_f32.shape
    assert J.dtype == np.float32


def test_recover_scene_radiance_t0_clamp_no_inf_nan():
    """Very low transmission (near zero) must not produce infinity or NaN after t0 clamping."""
    h, w = 32, 32
    img_f32 = np.full((h, w, 3), 0.5, dtype=np.float32)
    # Transmission nearly zero — without t0 clamping this would explode
    t = np.full((h, w), 1e-6, dtype=np.float32)
    A = np.array([0.8, 0.8, 0.8], dtype=np.float32)
    J = recover_scene_radiance(img_f32, t, A, t0=0.1)
    assert not np.isnan(J).any(), "NaN found in recovered radiance"
    assert not np.isinf(J).any(), "Inf found in recovered radiance"


# ---------------------------------------------------------------------------
# guided_filter_transmission tests
# ---------------------------------------------------------------------------

def test_guided_filter_transmission_shape_and_range():
    """Refined transmission must have same shape as input and values in [0, 1]."""
    h, w = 64, 64
    rng = np.random.default_rng(99)
    guide = rng.random((h, w)).astype(np.float32)
    raw_t = rng.random((h, w)).astype(np.float32)
    refined = guided_filter_transmission(guide, raw_t)
    assert refined.shape == (h, w)
    assert float(refined.min()) >= 0.0 - 1e-4
    assert float(refined.max()) <= 1.0 + 1e-4


# ---------------------------------------------------------------------------
# detect_clouds_simple with mixed image
# ---------------------------------------------------------------------------

def test_detect_clouds_simple_mixed_image_fraction():
    """Half-cloud / half-dark image should yield a cloud fraction between 0.1 and 0.9."""
    h, w = 64, 64
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # Top half: bright white — cloud
    img[: h // 2, :, :] = 230
    # Bottom half: dark vegetation — not cloud
    img[h // 2 :, :, :] = [25, 70, 25]
    mask = detect_clouds_simple(img)
    fraction = mask.mean()
    assert 0.1 < fraction < 0.9, f"Cloud fraction {fraction:.3f} not in expected range (0.1, 0.9)"
