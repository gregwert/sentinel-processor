"""Chip manifest CSV builder — geographic bounds and quality stats per chip."""
import csv
import io
import numpy as np


def chip_lat_lon_bounds(
    transform,
    crs,
    col_off: int,
    row_off: int,
    chip_w: int,
    chip_h: int,
) -> tuple:
    """Compute the geographic (WGS-84 lon/lat) bounding box for a chip window.

    Args:
        transform (affine.Affine): Source image affine transform (from rasterio metadata).
        crs (rasterio.crs.CRS or None): Coordinate reference system. When None, returns
            (None, None, None, None).
        col_off (int): Top-left pixel column offset of the chip in the source image.
        row_off (int): Top-left pixel row offset of the chip in the source image.
        chip_w (int): Chip width in pixels.
        chip_h (int): Chip height in pixels.

    Returns:
        tuple: (lon_min, lat_min, lon_max, lat_max) as floats, or
            (None, None, None, None) when CRS is None.
    """
    if crs is None:
        return (None, None, None, None)

    # Four corners in source CRS via affine multiplication — correctly handles
    # all six transform coefficients, including rotation terms b and d.
    corners_src = [
        transform * (col_off + dc, row_off + dr)
        for dc, dr in [(0, 0), (chip_w, 0), (0, chip_h), (chip_w, chip_h)]
    ]

    if crs.is_geographic:
        # Values are already lon/lat
        lons = [c[0] for c in corners_src]
        lats = [c[1] for c in corners_src]
    else:
        # Reproject projected coordinates to WGS-84
        from pyproj import Transformer
        transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        reprojected = [transformer.transform(x, y) for x, y in corners_src]
        lons = [c[0] for c in reprojected]
        lats = [c[1] for c in reprojected]

    return (min(lons), min(lats), max(lons), max(lats))


def build_manifest(
    grid,
    chip_stats: list | None = None,
    naming: str = "rowcol",
    fmt_ext: str = ".png",
) -> list:
    """Build a list of per-chip manifest row dicts.

    Args:
        grid (ChipGrid): The chip grid to build the manifest for.
        chip_stats (list[dict] or None): Output of apply_chip_filters's chip_stats list
            (keys: chip_index, cloud_pct, variance, rejected). When None, defaults to
            0.0 / False.
        naming (str): 'rowcol' or 'coords' — must match what export_chips uses.
        fmt_ext (str): File extension including dot, e.g. '.png'.

    Returns:
        list[dict]: One dict per chip with keys: chip_index, row, col, pixel_x_min,
            pixel_y_min, pixel_x_max, pixel_y_max, lon_min, lat_min, lon_max, lat_max,
            cloud_pct, variance, filename, rejected.
    """
    from sentinel_backend.chipping.gdal_chipper import get_chip
    from sentinel_backend.chipping.tile_exporter import _coords_filename

    # Build a stats lookup by chip_index
    stats_by_idx = {}
    if chip_stats:
        for s in chip_stats:
            stats_by_idx[s["chip_index"]] = s

    transform = grid.source_meta.get("transform")
    crs = grid.source_meta.get("crs")

    rows = []
    n_cols = grid.n_cols

    for idx, (col_off, row_off, chip_w, chip_h) in enumerate(grid.windows):
        row_idx = idx // n_cols
        col_idx = idx % n_cols

        # Pixel bounding box
        px_x_min = col_off
        px_y_min = row_off
        px_x_max = col_off + chip_w
        px_y_max = row_off + chip_h

        # Geographic bounding box
        lon_min, lat_min, lon_max, lat_max = chip_lat_lon_bounds(
            transform, crs, col_off, row_off, chip_w, chip_h
        )

        # Quality stats
        s = stats_by_idx.get(idx, {})
        cloud_pct = s.get("cloud_pct", 0.0)
        variance = s.get("variance", 0.0)
        rejected = s.get("rejected", False)

        # Filename — derive the same way tile_exporter does
        if naming == "coords" and transform is not None and crs is not None:
            _, chip_meta = get_chip(grid, idx)
            filename = _coords_filename(chip_meta, fmt_ext)
        else:
            filename = f"chip_r{row_idx:04d}_c{col_idx:04d}{fmt_ext}"

        rows.append({
            "chip_index": idx,
            "row": row_idx,
            "col": col_idx,
            "pixel_x_min": px_x_min,
            "pixel_y_min": px_y_min,
            "pixel_x_max": px_x_max,
            "pixel_y_max": px_y_max,
            "lon_min": f"{lon_min:.6f}" if lon_min is not None else "",
            "lat_min": f"{lat_min:.6f}" if lat_min is not None else "",
            "lon_max": f"{lon_max:.6f}" if lon_max is not None else "",
            "lat_max": f"{lat_max:.6f}" if lat_max is not None else "",
            "cloud_pct": f"{cloud_pct:.4f}",
            "variance": f"{variance:.2f}",
            "filename": filename,
            "rejected": str(rejected).lower(),
        })

    return rows


def write_manifest_csv(rows: list) -> bytes:
    """Serialise manifest row dicts to UTF-8 CSV bytes (no filesystem write).

    Args:
        rows (list[dict]): Output of build_manifest.

    Returns:
        bytes: UTF-8 encoded CSV suitable for zf.writestr() or st.download_button().
    """
    if not rows:
        return b""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue().encode("utf-8")
