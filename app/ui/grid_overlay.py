"""
Chip grid composite renderer — draws grid lines over the processed image
so users can see chip coverage before running chipping.
Also provides metres-to-pixels conversion using the rasterio geotransform.
"""
import math
from typing import Tuple

import numpy as np
from PIL import Image, ImageDraw

# Default dash/gap lengths (pixels) used by _dashed_line.
_DASH_LENGTH = 8
_GAP_LENGTH = 5

# Approximate metres per degree of latitude at the equator.
_METRES_PER_DEGREE_EQUATOR = 111320

# Minimum metres-per-degree value to avoid division by zero near the poles.
_MIN_METRES_PER_DEGREE = 1.0


def _dashed_line(draw, start: tuple, end: tuple, fill: tuple,
                 width: int = 1, dash: int = _DASH_LENGTH, gap: int = _GAP_LENGTH) -> None:
    """Draw a dashed line between start and end (horizontal or vertical only).

    Parameters
    ----------
    draw : PIL.ImageDraw.ImageDraw
        Active drawing context for the target image.
    start : tuple of int
        (x, y) pixel coordinate of the line's start point.
    end : tuple of int
        (x, y) pixel coordinate of the line's end point.
    fill : tuple of int
        RGB or RGBA colour used to draw each dash segment.
    width : int, optional
        Line width in pixels. Default 1.
    dash : int, optional
        Length of each drawn dash segment in pixels. Default 8.
    gap : int, optional
        Length of the gap between consecutive dash segments in pixels.
        Default 5.
    """
    x0, y0 = start
    x1, y1 = end
    if x0 == x1:  # vertical
        y = y0
        while y < y1:
            draw.line([(x0, y), (x0, min(y + dash - 1, y1))], fill=fill, width=width)
            y += dash + gap
    else:  # horizontal
        x = x0
        while x < x1:
            draw.line([(x, y0), (min(x + dash - 1, x1), y0)], fill=fill, width=width)
            x += dash + gap


def render_grid_composite(
    img_uint8: np.ndarray,
    windows: list,
    img_w: int,
    img_h: int,
    chip_w: int = 0,
    chip_h: int = 0,
    overlap: float = 0.0,
    line_colour: tuple = (255, 255, 0),
    dash_colour: tuple = (255, 165, 0),
    line_width: int = 1,
    max_display_px: int = 900,
) -> Image.Image:
    """Return PIL RGB image with chip grid lines drawn over it.

    When overlap > 0 and chip_w/chip_h are supplied, draws solid lines at chip
    start positions (the non-overlapping step grid) and orange dashed lines at
    chip end positions to show how far each chip extends into its neighbour.
    When overlap == 0, all window boundaries are drawn as solid lines.

    Parameters
    ----------
    img_uint8 : np.ndarray
        Shape (H, W, 3), dtype uint8. Source image to draw over.
    windows : list
        List of (col_off, row_off, w, h) tuples describing each chip window
        in source-image pixel coordinates. Used when overlap == 0.
    img_w : int
        Width of the source image in pixels.
    img_h : int
        Height of the source image in pixels.
    chip_w : int, optional
        Chip width in pixels. Required when overlap > 0. Default 0.
    chip_h : int, optional
        Chip height in pixels. Required when overlap > 0. Default 0.
    overlap : float, optional
        Fractional overlap between adjacent chips (0.0–1.0). Default 0.0.
    line_colour : tuple of int, optional
        RGB colour for solid grid lines. Default (255, 255, 0) is yellow.
    dash_colour : tuple of int, optional
        RGB colour for dashed overlap-extent lines. Default (255, 165, 0) is
        orange.
    line_width : int, optional
        Width of all drawn lines in pixels. Default 1.
    max_display_px : int, optional
        Long edge of the output image is capped at this pixel count for
        display performance. Default 900.

    Returns
    -------
    PIL.Image
        RGB image with grid lines composited on top, sized to fit
        max_display_px.
    """
    scale = min(max_display_px / max(img_h, img_w), 1.0)
    disp_w = max(int(img_w * scale), 1)
    disp_h = max(int(img_h * scale), 1)

    disp_img = Image.fromarray(img_uint8).resize((disp_w, disp_h), Image.LANCZOS).convert("RGB")
    draw = ImageDraw.Draw(disp_img)

    if chip_w > 0 and chip_h > 0:
        # Solid lines: the no-overlap reference grid (chip_w / chip_h spacing).
        # This is what the grid would look like with zero overlap and no edge clamping.
        no_overlap_cols = sorted(set(list(range(0, img_w, chip_w)) + [img_w]))
        no_overlap_rows = sorted(set(list(range(0, img_h, chip_h)) + [img_h]))
        no_overlap_cols_set = set(no_overlap_cols)
        no_overlap_rows_set = set(no_overlap_rows)

        for col in no_overlap_cols:
            x = min(int(col * scale), disp_w - 1)
            draw.line([(x, 0), (x, disp_h - 1)], fill=line_colour, width=line_width)
        for row in no_overlap_rows:
            y = min(int(row * scale), disp_h - 1)
            draw.line([(0, y), (disp_w - 1, y)], fill=line_colour, width=line_width)

        # Collect every actual chip boundary (left/right and top/bottom of each window).
        # Using the windows list directly handles both overlap fraction and edge clamping.
        all_col_bounds: set = set()
        all_row_bounds: set = set()
        for col_off, row_off, w, h in windows:
            all_col_bounds.add(col_off)
            all_col_bounds.add(col_off + w)
            all_row_bounds.add(row_off)
            all_row_bounds.add(row_off + h)

        # Dashed lines for any actual boundary not already on the solid grid.
        for col in sorted(all_col_bounds):
            if col not in no_overlap_cols_set and 0 < col < img_w:
                sx = min(int(col * scale), disp_w - 1)
                _dashed_line(draw, (sx, 0), (sx, disp_h - 1), dash_colour, width=line_width)
        for row in sorted(all_row_bounds):
            if row not in no_overlap_rows_set and 0 < row < img_h:
                sy = min(int(row * scale), disp_h - 1)
                _dashed_line(draw, (0, sy), (disp_w - 1, sy), dash_colour, width=line_width)

    else:
        # Fallback when chip dimensions are not provided: draw all boundaries solid.
        col_offs = sorted(set(col_off for col_off, _row_off, _w, _h in windows))
        row_offs = sorted(set(row_off for _col_off, row_off, _w, _h in windows))
        col_offs.append(img_w)
        row_offs.append(img_h)

        for col_off in col_offs:
            x = int(col_off * scale)
            draw.line([(x, 0), (x, disp_h - 1)], fill=line_colour, width=line_width)
        for row_off in row_offs:
            y = int(row_off * scale)
            draw.line([(0, y), (disp_w - 1, y)], fill=line_colour, width=line_width)

    return disp_img


def chip_size_metres_to_pixels(metres: float, source_meta: dict) -> Tuple[int, bool]:
    """Convert chip dimension from metres to pixels using rasterio geotransform.

    Parameters
    ----------
    metres : float
        Desired chip dimension in metres.
    source_meta : dict
        Rasterio dataset metadata dict. Must contain a ``"transform"`` key
        (an :class:`affine.Affine` instance). May contain an optional
        ``"crs"`` key (:class:`rasterio.crs.CRS`).

    Returns
    -------
    pixels : int
        Chip dimension converted to whole pixels (minimum 1).
    is_approximate : bool
        ``True`` when the conversion is only approximate — either because no
        CRS was available or because the CRS is geographic (degrees, not
        metres), in which case a cosine-corrected equatorial approximation is
        used.
    """
    transform = source_meta["transform"]
    crs = source_meta.get("crs")

    pixel_size_crs = abs(transform.a)

    if crs is None:
        pixels = max(1, round(metres / pixel_size_crs))
        return pixels, True

    if crs.is_projected:
        pixels = max(1, round(metres / pixel_size_crs))
        return pixels, False
    else:
        lat_centre = transform.f
        metres_per_degree = _METRES_PER_DEGREE_EQUATOR * math.cos(math.radians(abs(lat_centre)))
        metres_per_degree = max(metres_per_degree, _MIN_METRES_PER_DEGREE)
        pixel_size_m = pixel_size_crs * metres_per_degree
        pixels = max(1, round(metres / pixel_size_m))
        return pixels, True
