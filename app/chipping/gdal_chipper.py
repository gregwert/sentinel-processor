"""
chipping/gdal_chipper.py

Geospatially-aware image chipping using GDAL/rasterio primitives.
Computes a regular grid of chip windows over a processed image, supports
configurable overlap, and derives per-chip Affine geotransforms so that
exported chips carry correct spatial reference information.
"""

from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np
import rasterio.transform


@dataclass
class ChipGrid:
    windows: List[Tuple[int, int, int, int]]  # (col_off, row_off, w, h)
    source_image: np.ndarray                   # full processed uint8 (H,W,3)
    source_meta: dict                          # rasterio meta with CRS + transform
    chip_w: int
    chip_h: int
    n_rows: int
    n_cols: int
    edge_mode: str = "pad"

    @property
    def total(self):
        return len(self.windows)


def compute_chip_grid(img_w, img_h, chip_w, chip_h, overlap=0.0, edge_mode="pad") -> List[Tuple[int, int, int, int]]:
    """Return list of (col_off, row_off, w, h) windows covering the image."""
    step_x = max(int(chip_w * (1 - overlap)), 1)
    step_y = max(int(chip_h * (1 - overlap)), 1)

    if edge_mode == "overlap":
        windows = []
        col_offs = list(range(0, img_w, step_x))
        row_offs = list(range(0, img_h, step_y))
        seen = set()
        for row_off in row_offs:
            for col_off in col_offs:
                # Clamp so chip stays within image bounds — full chip size always
                actual_col = min(col_off, max(0, img_w - chip_w))
                actual_row = min(row_off, max(0, img_h - chip_h))
                window = (actual_col, actual_row, chip_w, chip_h)
                if window not in seen:
                    seen.add(window)
                    windows.append(window)
        return windows

    windows = []
    row = 0
    while row < img_h:
        col = 0
        while col < img_w:
            actual_w = min(chip_w, img_w - col)
            actual_h = min(chip_h, img_h - row)
            windows.append((col, row, actual_w, actual_h))
            col += step_x
        row += step_y

    return windows


def chip_affine(source_transform, col_off, row_off) -> rasterio.transform.Affine:
    """Return per-chip Affine geotransform shifted from source origin."""
    origin_x = source_transform.c + col_off * source_transform.a
    origin_y = source_transform.f + row_off * source_transform.e
    return rasterio.transform.Affine(
        source_transform.a, source_transform.b, origin_x,
        source_transform.d, source_transform.e, origin_y,
    )


def get_chip(grid: ChipGrid, index: int) -> Tuple[np.ndarray, dict]:
    """Return (chip_array uint8 HWC, chip_meta dict) for chip at flat index."""
    col_off, row_off, w, h = grid.windows[index]

    chip_raw = grid.source_image[row_off:row_off + h, col_off:col_off + w, :]

    # Pad to full chip size if this is an edge chip
    if chip_raw.shape[0] < grid.chip_h or chip_raw.shape[1] < grid.chip_w:
        padded = np.zeros(
            (grid.chip_h, grid.chip_w, grid.source_image.shape[2]), dtype=np.uint8
        )
        padded[:chip_raw.shape[0], :chip_raw.shape[1], :] = chip_raw
        chip_raw = padded

    # Build per-chip metadata
    chip_meta = grid.source_meta.copy()
    chip_meta.update({
        "height": grid.chip_h,
        "width": grid.chip_w,
        "count": grid.source_image.shape[2],
        "dtype": "uint8",
        "transform": chip_affine(grid.source_meta["transform"], col_off, row_off),
        "row_idx": index // grid.n_cols,
        "col_idx": index % grid.n_cols,
    })

    return chip_raw, chip_meta


def build_chip_grid(processed_img, source_meta, chip_w, chip_h, overlap=0.0, edge_mode="pad") -> ChipGrid:
    """Build ChipGrid from processed image; stores reference not copies."""
    H, W, C = processed_img.shape
    windows = compute_chip_grid(W, H, chip_w, chip_h, overlap, edge_mode)
    n_cols = len([wnd for wnd in windows if wnd[1] == 0])
    n_rows = len(set(wnd[1] for wnd in windows))
    return ChipGrid(
        windows=windows,
        source_image=processed_img,
        source_meta=source_meta,
        chip_w=chip_w,
        chip_h=chip_h,
        n_rows=n_rows,
        n_cols=n_cols,
        edge_mode=edge_mode,
    )
