"""
processing/dehazing.py

Dark Channel Prior (DCP) dehazing for Sentinel-2 imagery.
Implements the full He et al. DCP pipeline: dark channel computation,
atmospheric light estimation (with optional cloud-pixel exclusion),
transmission map estimation, guided-filter refinement, and scene
radiance recovery. Cloud pixels are detected and restored from the
original image after dehazing to avoid artefacts on bright surfaces.
"""

from dataclasses import dataclass

import numpy as np
import cv2


def detect_clouds_simple(img_uint8: np.ndarray, brightness_thresh: float = 0.75, saturation_thresh: float = 0.08) -> np.ndarray:
    """Produce a binary cloud mask from a uint8 RGB image using brightness and saturation thresholds.

    Args:
        img_uint8 (np.ndarray): Shape (H, W, 3), dtype uint8. Input RGB image.
        brightness_thresh (float, optional): Mean channel value (normalised 0-1) above which
            a pixel is considered a cloud candidate; default 0.75.
        saturation_thresh (float, optional): Max-minus-min channel spread (normalised 0-1)
            below which a pixel is considered spectrally neutral (white/grey), reinforcing
            the cloud classification; default 0.08.

    Returns:
        np.ndarray: Shape (H, W), dtype bool. True where a pixel is classified as cloud.
    """
    img_f32 = img_uint8.astype(np.float32) / 255.0
    brightness = img_f32.mean(axis=2)
    saturation = img_f32.max(axis=2) - img_f32.min(axis=2)
    cloud_mask = (brightness > brightness_thresh) & (saturation < saturation_thresh)
    return cloud_mask


def dark_channel(img_f32: np.ndarray, patch_size: int = 15) -> np.ndarray:
    """Compute the dark channel of a float32 RGB image using a minimum filter.

    Args:
        img_f32 (np.ndarray): Shape (H, W, 3), dtype float32, values in [0, 1]. Input
            image (or normalised radiance estimate).
        patch_size (int, optional): Side length of the rectangular erosion kernel used as
            the local neighbourhood window. Must be a positive odd integer; default 15.

    Returns:
        np.ndarray: Shape (H, W), dtype float32. Per-pixel minimum over all channels and
            all pixels within the ``patch_size`` neighbourhood.
    """
    min_channel = img_f32.min(axis=2).astype(np.float32)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    dark = cv2.erode(min_channel, kernel)
    return dark


def estimate_atmospheric_light(img_f32: np.ndarray, dark: np.ndarray, top_percent: float = 0.001, cloud_mask: np.ndarray = None) -> np.ndarray:
    """Estimate the global atmospheric light vector from the brightest dark-channel pixels.

    Args:
        img_f32 (np.ndarray): Shape (H, W, 3), dtype float32, values in [0, 1]. Input image.
        dark (np.ndarray): Shape (H, W), dtype float32. Pre-computed dark channel of
            ``img_f32``.
        top_percent (float, optional): Fraction of non-cloud pixels with the highest
            dark-channel values used as atmospheric light candidates; default 0.001.
        cloud_mask (np.ndarray or None, optional): Shape (H, W), dtype bool. If provided,
            cloud pixels are excluded from candidate selection. Falls back to all pixels
            when more than 80 % of the scene is flagged as cloud; default None.

    Returns:
        np.ndarray: Shape (3,), dtype float32. RGB atmospheric light estimate A, taken as
            the candidate pixel with the highest sum across channels.
    """
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


def transmission_map(img_f32: np.ndarray, A: np.ndarray, patch_size: int = 15, omega: float = 0.95) -> np.ndarray:
    """Estimate the transmission map from the normalised dark channel.

    Args:
        img_f32 (np.ndarray): Shape (H, W, 3), dtype float32, values in [0, 1]. Input
            hazy image.
        A (np.ndarray): Shape (3,), dtype float32. Atmospheric light estimate.
        patch_size (int, optional): Patch size forwarded to :func:`dark_channel`; default 15.
        omega (float, optional): Haze retention factor in [0, 1]. Values close to 1 remove
            more haze but may over-darken already dark regions; default 0.95.

    Returns:
        np.ndarray: Shape (H, W), dtype float32. Raw (unrefined) transmission map with
            values in (0, 1].
    """
    A_broadcast = A[np.newaxis, np.newaxis, :]
    normalized = img_f32 / (A_broadcast + 1e-6)
    dark = dark_channel(normalized.astype(np.float32), patch_size)
    t = 1.0 - omega * dark
    return t.astype(np.float32)


def guided_filter_transmission(guide: np.ndarray, transmission: np.ndarray, radius: int = 60, eps: float = 1e-3) -> np.ndarray:
    """Refine the transmission map with a guided filter to reduce halo artefacts.

    Args:
        guide (np.ndarray): Shape (H, W), dtype float32, values in [0, 1]. Greyscale
            guidance image (typically the luminance channel of the hazy input).
        transmission (np.ndarray): Shape (H, W), dtype float32, values in [0, 1]. Raw
            transmission map to be refined.
        radius (int, optional): Guided-filter window radius in pixels. Larger values produce
            smoother results at the cost of stronger halo suppression; default 60.
        eps (float, optional): Regularisation parameter controlling the degree of edge
            preservation. Smaller values preserve more edges; default 1e-3.

    Returns:
        np.ndarray: Shape (H, W), dtype float32. Edge-preserving refined transmission map.
            Falls back to bilateral filtering when ``cv2.ximgproc`` is unavailable.
    """
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


def recover_scene_radiance(img_f32: np.ndarray, t: np.ndarray, A: np.ndarray, t0: float = 0.1) -> np.ndarray:
    """Recover the dehazed scene radiance from the hazy image, transmission, and atmospheric light.

    Args:
        img_f32 (np.ndarray): Shape (H, W, 3), dtype float32, values in [0, 1]. Hazy input
            image.
        t (np.ndarray): Shape (H, W), dtype float32. Refined transmission map.
        A (np.ndarray): Shape (3,), dtype float32. Atmospheric light estimate.
        t0 (float, optional): Lower bound on transmission to avoid division by near-zero values
            and excessive noise amplification in very dense haze; default 0.1.

    Returns:
        np.ndarray: Shape (H, W, 3), dtype float32, values clipped to [0, 1]. Recovered
            scene radiance J computed as ``(I - A) / max(t, t0) + A``.
    """
    t_clamped = np.maximum(t[:, :, np.newaxis], t0)
    J = (img_f32 - A) / t_clamped + A
    return np.clip(J, 0, 1).astype(np.float32)


@dataclass
class Dehazer:
    """Dark Channel Prior dehazing with optional cloud-adaptive atmospheric light.

    Attributes:
        patch_size (int): Side length of the local minimum-filter patch used for dark channel
            computation and transmission estimation. Must be a positive odd integer; default 15.
        omega (float): Haze retention factor. Values close to 1 remove more haze but risk
            over-darkening already dark pixels; default 0.95.
        t0 (float): Minimum transmission clamp applied before scene radiance recovery to
            prevent division-by-zero and noise amplification; default 0.1.
        use_guided_filter (bool): When True the raw transmission map is refined by a guided
            (or bilateral) filter to suppress halo artefacts at depth discontinuities;
            default True.
        mask_clouds (bool): When True cloud pixels are detected before dehazing and restored
            from the original image afterwards to prevent artefacts on bright surfaces;
            default True.
        brightness_thresh (float): Mean normalised brightness threshold forwarded to
            :func:`detect_clouds_simple`; default 0.75.
        saturation_thresh (float): Max-minus-min normalised saturation threshold forwarded to
            :func:`detect_clouds_simple`; default 0.08.
    """

    patch_size: int = 15
    omega: float = 0.95
    t0: float = 0.1
    use_guided_filter: bool = True
    mask_clouds: bool = True
    brightness_thresh: float = 0.75
    saturation_thresh: float = 0.08

    def run(self, img: np.ndarray) -> np.ndarray:
        """Run the full DCP pipeline on ``img`` and return the dehazed image.

        Args:
            img (np.ndarray): Shape (H, W, 3), dtype uint8. Hazy input RGB image.

        Returns:
            np.ndarray: Shape (H, W, 3), dtype uint8. Dehazed image with cloud pixels
                restored from the original when ``mask_clouds`` is True.
        """
        img_f32 = img.astype(np.float32) / 255.0

        cloud_mask = None
        if self.mask_clouds:
            cloud_mask = detect_clouds_simple(
                img, self.brightness_thresh, self.saturation_thresh
            )

        dark = dark_channel(img_f32, self.patch_size)
        A = estimate_atmospheric_light(img_f32, dark, cloud_mask=cloud_mask)
        t = transmission_map(img_f32, A, self.patch_size, self.omega)

        if self.use_guided_filter:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
            t = guided_filter_transmission(gray, t)

        J = recover_scene_radiance(img_f32, t, A, self.t0)
        result = (J * 255).astype(np.uint8)

        # CRITICAL: restore original cloud pixels AFTER guided filter refinement
        if cloud_mask is not None:
            result[cloud_mask] = img[cloud_mask]

        return result


def dehaze(
    img_uint8: np.ndarray,
    patch_size: int = 15,
    omega: float = 0.95,
    t0: float = 0.1,
    use_guided_filter: bool = True,
    mask_clouds: bool = True,
    brightness_thresh: float = 0.75,
    saturation_thresh: float = 0.08,
) -> np.ndarray:
    """Convenience wrapper — constructs a Dehazer and calls run().

    Args:
        img_uint8 (np.ndarray): Shape (H, W, 3), dtype uint8. Hazy input RGB image.
        patch_size (int, optional): Forwarded to :class:`Dehazer`; default 15.
        omega (float, optional): Forwarded to :class:`Dehazer`; default 0.95.
        t0 (float, optional): Forwarded to :class:`Dehazer`; default 0.1.
        use_guided_filter (bool, optional): Forwarded to :class:`Dehazer`; default True.
        mask_clouds (bool, optional): Forwarded to :class:`Dehazer`; default True.
        brightness_thresh (float, optional): Forwarded to :class:`Dehazer`; default 0.75.
        saturation_thresh (float, optional): Forwarded to :class:`Dehazer`; default 0.08.

    Returns:
        np.ndarray: Shape (H, W, 3), dtype uint8. Dehazed image returned by
            :meth:`Dehazer.run`.
    """
    return Dehazer(
        patch_size=patch_size,
        omega=omega,
        t0=t0,
        use_guided_filter=use_guided_filter,
        mask_clouds=mask_clouds,
        brightness_thresh=brightness_thresh,
        saturation_thresh=saturation_thresh,
    ).run(img_uint8)
