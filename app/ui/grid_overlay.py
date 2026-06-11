"""
Chip grid composite renderer — draws grid lines over the processed image
so users can see chip coverage before running chipping.
Also provides metres-to-pixels conversion using the rasterio geotransform.
"""
import math
from typing import Tuple

import numpy as np
from PIL import Image, ImageDraw
from rasterio.crs import CRS


def _dashed_v(draw, x, img_h, fill, width=1, dash=8, gap=5):
    """Draw a vertical dashed line at pixel x."""
    y = 0
    while y < img_h:
        draw.line([(x, y), (x, min(y + dash - 1, img_h - 1))], fill=fill, width=width)
        y += dash + gap


def _dashed_h(draw, y, img_w, fill, width=1, dash=8, gap=5):
    """Draw a horizontal dashed line at pixel y."""
    x = 0
    while x < img_w:
        draw.line([(x, y), (min(x + dash - 1, img_w - 1), y)], fill=fill, width=width)
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
    """
    scale = min(max_display_px / max(img_h, img_w), 1.0)
    disp_w = max(int(img_w * scale), 1)
    disp_h = max(int(img_h * scale), 1)

    disp_img = Image.fromarray(img_uint8).resize((disp_w, disp_h), Image.LANCZOS).convert("RGB")
    draw = ImageDraw.Draw(disp_img)

    if overlap > 0 and chip_w > 0 and chip_h > 0:
        step_x = max(int(chip_w * (1 - overlap)), 1)
        step_y = max(int(chip_h * (1 - overlap)), 1)

        col_starts = list(range(0, img_w, step_x))
        row_starts = list(range(0, img_h, step_y))
        col_starts_set = set(col_starts)
        row_starts_set = set(row_starts)

        # Solid lines at chip start positions
        for col_off in col_starts:
            x = int(col_off * scale)
            draw.line([(x, 0), (x, disp_h - 1)], fill=line_colour, width=line_width)
        for row_off in row_starts:
            y = int(row_off * scale)
            draw.line([(0, y), (disp_w - 1, y)], fill=line_colour, width=line_width)

        # Solid image boundary
        draw.line([(disp_w - 1, 0), (disp_w - 1, disp_h - 1)], fill=line_colour, width=line_width)
        draw.line([(0, disp_h - 1), (disp_w - 1, disp_h - 1)], fill=line_colour, width=line_width)

        # Dashed lines at chip end positions (showing overlap extent)
        for col_off in col_starts:
            end_x = col_off + chip_w
            if 0 < end_x < img_w and end_x not in col_starts_set:
                _dashed_v(draw, int(end_x * scale), disp_h, dash_colour, width=line_width)
        for row_off in row_starts:
            end_y = row_off + chip_h
            if 0 < end_y < img_h and end_y not in row_starts_set:
                _dashed_h(draw, int(end_y * scale), disp_w, dash_colour, width=line_width)

    else:
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
    """
    Convert chip dimension from metres to pixels using rasterio geotransform.
    Returns (pixels: int, is_approximate: bool).
    is_approximate=True when CRS is geographic (degrees not metres).
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
        metres_per_degree = 111320 * math.cos(math.radians(abs(lat_centre)))
        metres_per_degree = max(metres_per_degree, 1.0)
        pixel_size_m = pixel_size_crs * metres_per_degree
        pixels = max(1, round(metres / pixel_size_m))
        return pixels, True
