"""
processing/dehazing.py

Dark Channel Prior (DCP) dehazing for Sentinel-2 imagery.
Implements the full He et al. DCP pipeline: dark channel computation,
atmospheric light estimation (with optional cloud-pixel exclusion),
transmission map estimation, guided-filter refinement, and scene
radiance recovery. Cloud pixels are detected and restored from the
original image after dehazing to avoid artefacts on bright surfaces.
"""

import numpy as np
import cv2


def detect_clouds_simple(img_uint8, brightness_thresh=0.75, saturation_thresh=0.08):
    """Return bool mask (H,W); True = cloud pixel."""
    img_f32 = img_uint8.astype(np.float32) / 255.0
    brightness = img_f32.mean(axis=2)
    saturation = img_f32.max(axis=2) - img_f32.min(axis=2)
    cloud_mask = (brightness > brightness_thresh) & (saturation < saturation_thresh)
    return cloud_mask


def dark_channel(img_f32, patch_size=15):
    """Return dark channel map float32 (H,W) from float32 (H,W,3) image in [0,1]."""
    min_channel = img_f32.min(axis=2).astype(np.float32)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    dark = cv2.erode(min_channel, kernel)
    return dark


def estimate_atmospheric_light(img_f32, dark, top_percent=0.001, cloud_mask=None):
    """Return atmospheric light A as float32 (3,), excluding cloud pixels from candidates."""
    h, w, c = img_f32.shape
    total_pixels = h * w

    img_flat = img_f32.reshape(-1, c)
    dark_flat = dark.flatten()

    if cloud_mask is not None:
        valid_indices = np.where(~cloud_mask.flatten())[0]
        # Fall back to all pixels if scene is too cloudy (>80% cloud)
        if len(valid_indices) < total_pixels * 0.2:
            valid_indices = np.arange(total_pixels)
    else:
        valid_indices = np.arange(total_pixels)

    valid_dark = dark_flat[valid_indices]
    valid_img = img_flat[valid_indices]

    n_bright = max(int(len(valid_indices) * top_percent), 1)
    indices = np.argpartition(valid_dark, -n_bright)[-n_bright:]
    candidates = valid_img[indices]

    # Pick the candidate with the highest sum across channels
    best = np.argmax(candidates.sum(axis=1))
    A = candidates[best].astype(np.float32)
    return A


def transmission_map(img_f32, A, patch_size=15, omega=0.95):
    """Return transmission map float32 (H,W)."""
    A_broadcast = A[np.newaxis, np.newaxis, :]
    normalized = img_f32 / (A_broadcast + 1e-6)
    dark = dark_channel(normalized.astype(np.float32), patch_size)
    t = 1.0 - omega * dark
    return t.astype(np.float32)


def guided_filter_transmission(guide, transmission, radius=60, eps=1e-3):
    """Refine transmission map using guided filter to reduce halo artifacts."""
    guide8 = (guide * 255).astype(np.uint8)
    t8 = (transmission * 255).astype(np.uint8)

    try:
        refined8 = cv2.ximgproc.guidedFilter(
            guide8, t8, radius, eps * 255 * 255
        )
        refined = refined8.astype(np.float32) / 255.0
    except (ImportError, AttributeError):
        d = max(radius // 4, 5)
        refined8 = cv2.bilateralFilter(t8, d=d, sigmaColor=75, sigmaSpace=75)
        refined = refined8.astype(np.float32) / 255.0

    return refined


def recover_scene_radiance(img_f32, t, A, t0=0.1):
    """Recover dehazed image using J(x) = (I(x) - A) / max(t(x), t0) + A."""
    t_clamped = np.maximum(t[:, :, np.newaxis], t0)
    J = (img_f32 - A) / t_clamped + A
    return np.clip(J, 0, 1).astype(np.float32)


def dehaze(
    img_uint8,
    patch_size=15,
    omega=0.95,
    t0=0.1,
    use_guided_filter=True,
    mask_clouds=True,
    brightness_thresh=0.75,
    saturation_thresh=0.08,
):
    """Full DCP dehazing pipeline; restores cloud pixels from original after dehazing."""
    img_f32 = img_uint8.astype(np.float32) / 255.0

    cloud_mask = None
    if mask_clouds:
        cloud_mask = detect_clouds_simple(
            img_uint8, brightness_thresh, saturation_thresh
        )

    dark = dark_channel(img_f32, patch_size)
    A = estimate_atmospheric_light(img_f32, dark, cloud_mask=cloud_mask)
    t = transmission_map(img_f32, A, patch_size, omega)

    if use_guided_filter:
        gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        t = guided_filter_transmission(gray, t)

    J = recover_scene_radiance(img_f32, t, A, t0)
    result = (J * 255).astype(np.uint8)

    # CRITICAL: restore original cloud pixels AFTER guided filter refinement
    if cloud_mask is not None:
        result[cloud_mask] = img_uint8[cloud_mask]

    return result
