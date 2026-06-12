"""
chipping/gdal_chipper.py

Geospatially-aware image chipping using GDAL/rasterio primitives.
Computes a regular grid of chip windows over a processed image, supports
configurable overlap, and derives per-chip Affine geotransforms so that
exported chips carry correct spatial reference information.
"""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import rasterio.transform


@dataclass
class ChipGrid:
    """Holds all state needed to iterate and extract chips from one source image.

    Attributes:
        windows (List[Tuple[int, int, int, int]]): Ordered list of (col_off, row_off, w, h)
            pixel windows, one per chip.
        source_image (np.ndarray): Full processed uint8 image array of shape (H, W, 3).
            Stored by reference — not copied — so the array must remain alive for the
            lifetime of this grid.
        source_meta (dict): Rasterio-style metadata dict containing at minimum 'crs' and
            'transform' keys for the source image.
        chip_w (int): Nominal chip width in pixels. Edge chips may be narrower before
            zero-padding is applied by get_chip.
        chip_h (int): Nominal chip height in pixels. Edge chips may be shorter before
            zero-padding is applied by get_chip.
        n_rows (int): Number of chip rows in the grid.
        n_cols (int): Number of chip columns in the grid.
        edge_mode (str): Strategy used when the image extent is not an exact multiple of
            the chip size. Either 'pad' or 'overlap' (see compute_chip_grid).
    """

    windows: List[Tuple[int, int, int, int]]  # (col_off, row_off, w, h) per chip
    source_image: np.ndarray                   # full processed uint8 array (H, W, 3)
    source_meta: dict                          # rasterio meta dict with CRS + transform
    chip_w: int                                # nominal chip width in pixels
    chip_h: int                                # nominal chip height in pixels
    n_rows: int                                # number of chip rows in the grid
    n_cols: int                                # number of chip columns in the grid
    edge_mode: str = "pad"                     # 'pad' or 'overlap' edge handling

    @property
    def total(self) -> int:
        return len(self.windows)


def compute_chip_grid(img_w, img_h, chip_w, chip_h, overlap=0.0, edge_mode="pad") -> List[Tuple[int, int, int, int]]:
    """Return a list of (col_off, row_off, w, h) windows that tile the image.

    Args:
        img_w (int): Width of the source image in pixels.
        img_h (int): Height of the source image in pixels.
        chip_w (int): Desired chip width in pixels.
        chip_h (int): Desired chip height in pixels.
        overlap (float, optional): Fractional overlap between adjacent chips in the range
            [0, 1). An overlap of 0.25 means chips share 25 % of their width/height with
            their neighbours. Default is 0.0 (no overlap).
        edge_mode (str, optional): Controls how chips at the image boundary are handled.

            'pad': The final row/column of chips may be smaller than chip_w/chip_h. The
            caller (get_chip) zero-pads them to the nominal size. This preserves exact
            coverage with no repeated pixels.

            'overlap': The last chip in each row/column is shifted left/up so that it fits
            entirely within the image. This avoids sub-size chips at the cost of introducing
            extra overlap at the boundary.

            Default is 'pad'.

    Returns:
        List[Tuple[int, int, int, int]]: Ordered list of (col_off, row_off, w, h) pixel
            windows, traversed row-major (left-to-right, top-to-bottom).
    """
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
    """Return an Affine geotransform whose origin is the top-left of a chip.

    The chip origin is derived by walking col_off pixel steps in X and
    row_off pixel steps in Y from the source image origin.  Specifically:

        origin_x = source_transform.c + col_off * source_transform.a
        origin_y = source_transform.f + row_off * source_transform.e

    where ``.c`` / ``.f`` are the source image's top-left corner coordinates
    (easting/northing or longitude/latitude) and ``.a`` / ``.e`` are the
    pixel width and pixel height (typically negative for north-up images).
    The rotation coefficients ``.b`` and ``.d`` are copied unchanged from
    the source transform so that non-north-up imagery is handled correctly.

    Args:
        source_transform (rasterio.transform.Affine): Affine geotransform of the full
            source image.
        col_off (int): Pixel column offset of the chip's top-left corner within the
            source image.
        row_off (int): Pixel row offset of the chip's top-left corner within the source
            image.

    Returns:
        rasterio.transform.Affine: Affine geotransform placing the chip's top-left pixel
            at the correct geographic coordinate.
    """
    origin_x = source_transform.c + col_off * source_transform.a
    origin_y = source_transform.f + row_off * source_transform.e
    return rasterio.transform.Affine(
        source_transform.a, source_transform.b, origin_x,
        source_transform.d, source_transform.e, origin_y,
    )


def get_chip(grid: ChipGrid, index: int) -> Tuple[np.ndarray, dict]:
    """Return the chip array and metadata for the chip at a flat grid index.

    Chips that fall on the right or bottom edge of the image may be smaller
    than the nominal chip size.  In that case the extracted region is
    zero-padded (top-left aligned) to exactly (chip_h, chip_w, C) so that
    every chip returned by this function has a consistent shape.

    Args:
        grid (ChipGrid): The chip grid produced by build_chip_grid. Holds the source image
            reference, window list, and spatial metadata.
        index (int): Flat (row-major) index into grid.windows. Must satisfy
            0 <= index < grid.total.

    Returns:
        chip_array (np.ndarray): uint8 array of shape (chip_h, chip_w, C) in HWC channel
            order. Edge chips are zero-padded to reach the nominal size.
        chip_meta (dict): Copy of grid.source_meta updated with chip-specific values:
            'height', 'width', 'count', 'dtype', 'transform', 'row_idx', 'col_idx'.
    """
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
    """Build a ChipGrid from a processed image array.

    ``processed_img`` is stored by reference inside the returned ChipGrid,
    not copied. The array must remain alive and unmodified for the lifetime
    of the grid and any chips extracted from it.

    Args:
        processed_img (np.ndarray): Processed source image of shape (H, W, C), dtype uint8.
        source_meta (dict): Rasterio-style metadata dict for the source image, containing at
            minimum 'crs' and 'transform' keys.
        chip_w (int): Desired chip width in pixels.
        chip_h (int): Desired chip height in pixels.
        overlap (float, optional): Fractional overlap between adjacent chips; passed through
            to compute_chip_grid. Default is 0.0.
        edge_mode (str, optional): Edge-handling strategy ('pad' or 'overlap'); passed
            through to compute_chip_grid. Default is 'pad'.

    Returns:
        ChipGrid: Populated grid whose windows, row/column counts, and source references are
            ready for use with get_chip or export_chips.
    """
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
