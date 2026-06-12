"""
Cloud mask composite renderer for real-time threshold feedback.
Overlays a semi-transparent colour mask on the stretched image to show
which pixels are classified as cloud by detect_clouds_simple().
"""
import numpy as np
from PIL import Image


def render_cloud_composite(
    img_uint8: np.ndarray,
    cloud_mask: np.ndarray,
    overlay_colour: tuple = (255, 0, 0),
    alpha: float = 0.45,
    max_display_px: int = 800,
) -> Image.Image:
    """Render a semi-transparent cloud mask overlay on the source image.

    Args:
        img_uint8 (np.ndarray): Shape (H, W, 3), dtype uint8. Source image to overlay on.
        cloud_mask (np.ndarray): Shape (H, W), dtype bool. True where cloud pixels are
            detected.
        overlay_colour (tuple of int, optional): RGB colour for the cloud overlay.
            Default (255, 0, 0) is red.
        alpha (float, optional): Opacity of the cloud overlay layer. 0.0 = invisible,
            1.0 = opaque.
        max_display_px (int, optional): Long edge of the output image is capped at this
            pixel count for display performance. Default 800.

    Returns:
        PIL.Image: RGBA composite image ready for st.image().
    """
    h, w = img_uint8.shape[:2]
    scale = min(max_display_px / max(h, w), 1.0)
    disp_w = max(int(w * scale), 1)
    disp_h = max(int(h * scale), 1)

    # Downsample base image
    base_pil = Image.fromarray(img_uint8).resize((disp_w, disp_h), Image.LANCZOS)

    # Downsample cloud mask: bool -> uint8, resize with NEAREST, back to bool
    mask_uint8 = (cloud_mask.astype(np.uint8) * 255)
    mask_pil = Image.fromarray(mask_uint8).resize((disp_w, disp_h), Image.NEAREST)
    mask_downsampled = np.array(mask_pil) > 0

    # Build RGBA composite
    base_rgba = base_pil.convert("RGBA")

    # Overlay layer: solid colour with alpha, same size as display image
    overlay_layer = Image.new(
        "RGBA", (disp_w, disp_h), (*overlay_colour, int(alpha * 255))
    )

    # Mask layer: 255 where cloud, 0 elsewhere
    mask_arr = np.where(mask_downsampled, 255, 0).astype(np.uint8)

    # Paste overlay onto base using the cloud mask
    base_rgba.paste(overlay_layer, mask=Image.fromarray(mask_arr))

    return base_rgba


def compute_cloud_stats(cloud_mask: np.ndarray) -> dict:
    """Compute summary statistics for a cloud detection mask.

    Args:
        cloud_mask (np.ndarray): Shape (H, W), dtype bool. True where cloud pixels are
            detected.

    Returns:
        dict: Dictionary with keys:
            ``cloud_pct`` (float): percentage of pixels classified as cloud, rounded to
            one decimal place.
            ``cloud_px`` (int): absolute count of cloud pixels.
            ``total_px`` (int): total pixel count in the mask.
    """
    total_px = cloud_mask.size
    cloud_px = int(cloud_mask.sum())
    cloud_pct = round(cloud_px / total_px * 100, 1)
    return {"cloud_pct": cloud_pct, "cloud_px": cloud_px, "total_px": total_px}
