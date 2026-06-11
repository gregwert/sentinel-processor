"""
chipping/tile_exporter.py

Parallel chip export to disk and in-memory ZIP packaging.
Writes individual chip files (PNG or GeoTIFF with embedded CRS) using a
ProcessPoolExecutor for throughput, then offers ZIP bundling of all
exported paths for direct download via Streamlit's download_button.
"""

import os
import io
import zipfile
import concurrent.futures
import multiprocessing
from dataclasses import dataclass
from typing import List

import numpy as np
from PIL import Image


@dataclass
class ExportConfig:
    """Encapsulates all chip export options so call sites pass one object.

    Fields
    ------
    fmt : str
        Output format. One of 'png', 'jpeg', 'geotiff', 'npy'.
    naming : str
        Chip filename scheme. 'rowcol' → r0001_c0002; 'coords' → lat/lon or easting/northing.
    normalise : bool
        If True, apply z-score normalisation using global_stats before export.
    global_stats : dict or None
        Must be provided when normalise=True. Dict with keys 'mean' and 'std',
        each a list of 3 floats (one per channel).
    """

    fmt: str = "png"                  # output file format: 'png', 'jpeg', 'geotiff', or 'npy'
    naming: str = "coords"            # filename scheme: 'rowcol' or 'coords'
    normalise: bool = False           # whether to apply z-score normalisation before export
    global_stats: dict = None         # dict with 'mean' and 'std' lists (3 floats each) for normalisation


def _export_single_chip(args) -> str:
    """Write one chip to disk; designed as a top-level picklable worker function.

    Accepts a single tuple argument so it can be used directly with
    executor.map without a lambda (which is not picklable).

    Parameters
    ----------
    args : tuple
        Seven-element tuple of:
        ``(chip_array, chip_meta, out_path, fmt, normalise, global_mean, global_std)``

        chip_array : np.ndarray
            uint8 HWC chip array.
        chip_meta : dict
            Chip metadata dict as returned by get_chip; used for GeoTIFF
            spatial reference fields.
        out_path : str
            Absolute path to write the output file to.
        fmt : str
            One of 'png', 'jpeg', 'geotiff', 'npy'.
        normalise : bool
            If True, apply z-score normalisation before writing.
        global_mean : list or None
            Per-channel mean values (3 floats) for z-score normalisation.
        global_std : list or None
            Per-channel standard deviation values (3 floats) for z-score
            normalisation.

    Returns
    -------
    str
        The out_path that was written, enabling the caller to collect all
        paths from executor.map results.
    """
    chip_array, chip_meta, out_path, fmt, normalise, global_mean, global_std = args

    if normalise and global_mean is not None and global_std is not None:
        mean = np.array(global_mean, dtype=np.float32)
        std = np.array(global_std, dtype=np.float32) + 1e-6
        chip_f = chip_array.astype(np.float32)
        normalised = (chip_f - mean) / std
        if fmt == "npy":
            # NPY gets raw float32 — do NOT clip to uint8
            # Save normalised float array instead of chip_array
            np.save(out_path, normalised)
            return out_path
        else:
            # Rescale normalised float back to uint8 for visual formats
            rescaled = normalised * 45.0 + 127.5  # approximate target distribution
            chip_array = np.clip(rescaled, 0, 255).astype(np.uint8)

    if fmt == "png":
        img = Image.fromarray(chip_array.astype(np.uint8), mode="RGB")
        img.save(out_path, format="PNG")

    elif fmt == "jpeg":
        img = Image.fromarray(chip_array.astype(np.uint8), mode="RGB")
        img.save(out_path, format="JPEG", quality=90)

    elif fmt == "geotiff":
        import rasterio
        with rasterio.open(
            out_path,
            "w",
            driver="GTiff",
            height=chip_meta["height"],
            width=chip_meta["width"],
            count=chip_meta["count"],
            dtype="uint8",
            crs=chip_meta["crs"],
            transform=chip_meta["transform"],
            compress="lzw",
        ) as dst:
            # chip_array is HWC; rasterio expects bands-first (CHW)
            for band_idx in range(chip_meta["count"]):
                dst.write(chip_array[:, :, band_idx], band_idx + 1)

    elif fmt == "npy":
        np.save(out_path, chip_array)

    return out_path


def _coords_filename(chip_meta: dict, ext: str) -> str:
    """Build a filesystem-safe chip filename from the chip's geotransform origin.

    For geographic CRS (longitude/latitude) the filename encodes the
    latitude and longitude of the chip's top-left corner with decimal
    points replaced by 'p' to avoid OS path issues.  For projected CRS
    (e.g. UTM) integer metre values are used instead.

    Parameters
    ----------
    chip_meta : dict
        Chip metadata dict as returned by get_chip.  Must contain a
        'transform' key (rasterio Affine) and an optional 'crs' key
        (rasterio CRS).
    ext : str
        File extension including the leading dot, e.g. '.png' or '.tif'.

    Returns
    -------
    str
        Filename string such as 'chip_51p4823N_000p1034W.png' (geographic)
        or 'chip_5714000N_341000E.tif' (projected).
    """
    t = chip_meta["transform"]
    crs = chip_meta.get("crs")
    if crs is not None and crs.is_geographic:
        lat, lon = t.f, t.c
        lat_s = f"{abs(lat):.4f}".replace(".", "p")
        lon_s = f"{abs(lon):.4f}".replace(".", "p")
        lat_dir = "N" if lat >= 0 else "S"
        lon_dir = "E" if lon >= 0 else "W"
        return f"chip_{lat_s}{lat_dir}_{lon_s}{lon_dir}{ext}"
    else:
        # Projected CRS (UTM etc) — integer metres, no decimal noise
        north = int(round(t.f))
        east = int(round(t.c))
        north_dir = "N" if north >= 0 else "S"
        east_dir = "E" if east >= 0 else "W"
        return f"chip_{abs(north)}{north_dir}_{abs(east)}{east_dir}{ext}"


def export_chips(grid, output_dir: str, fmt: str = "png", naming: str = "rowcol",
                 normalise: bool = False, global_stats=None,
                 config: "ExportConfig | None" = None) -> List[str]:
    """Export all chips in a ChipGrid to disk using a process pool.

    Individual keyword arguments (``fmt``, ``naming``, ``normalise``,
    ``global_stats``) are provided for backward compatibility.  When
    ``config`` is given it takes precedence over those arguments, so
    existing call sites that pass individual kwargs continue to work
    without any changes.

    Parameters
    ----------
    grid : ChipGrid
        The chip grid produced by build_chip_grid. Provides window list,
        source image reference, and spatial metadata.
    output_dir : str
        Directory to write chip files into. Created if it does not exist.
    fmt : str, optional
        Output format. One of 'png', 'jpeg', 'geotiff', 'npy'.
        Default is 'png'.
    naming : str, optional
        Chip filename scheme. 'rowcol' produces names like
        'chip_r0001_c0002.png'; 'coords' derives names from the chip's
        geographic origin via _coords_filename. Default is 'rowcol'.
    normalise : bool, optional
        If True, apply z-score normalisation using global_stats before
        writing. Default is False.
    global_stats : dict or None, optional
        Required when normalise=True. Dict with keys 'mean' and 'std',
        each a list of 3 floats (one per channel). Default is None.
    config : ExportConfig or None, optional
        If provided, its values override fmt, naming, normalise, and
        global_stats. Default is None.

    Returns
    -------
    List[str]
        Absolute paths of all written chip files, in window order
        (row-major, matching grid.windows).
    """
    if config is not None:
        fmt = config.fmt
        naming = config.naming
        normalise = config.normalise
        global_stats = config.global_stats
    os.makedirs(output_dir, exist_ok=True)

    ext_map = {
        "png": ".png",
        "jpeg": ".jpg",
        "geotiff": ".tif",
        "npy": ".npy",
    }
    ext = ext_map.get(fmt, ".png")

    # Import here to avoid circular import issues at module level
    from chipping.gdal_chipper import get_chip

    global_mean = global_stats.get("mean") if global_stats else None
    global_std = global_stats.get("std") if global_stats else None

    args_list = []
    for index in range(grid.total):
        chip_array, chip_meta = get_chip(grid, index)
        row_idx = chip_meta["row_idx"]
        col_idx = chip_meta["col_idx"]

        if naming == "rowcol":
            fname = f"chip_r{row_idx:04d}_c{col_idx:04d}{ext}"
        elif naming == "coords":
            fname = _coords_filename(chip_meta, ext)
        else:
            fname = f"chip_r{row_idx:04d}_c{col_idx:04d}{ext}"

        out_path = os.path.join(output_dir, fname)
        args_list.append((chip_array, chip_meta, out_path, fmt, normalise, global_mean, global_std))

    workers = os.cpu_count() or 4
    ctx = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as executor:
        results = list(executor.map(_export_single_chip, args_list))

    return results


def zip_export(chip_paths: List[str]) -> bytes:
    """Bundle a list of chip files into an in-memory ZIP archive.

    Intended for use with Streamlit's ``st.download_button`` so that a full
    chip set can be offered as a single download without writing a ZIP to
    disk.  Each chip is stored under its bare filename (no directory path)
    inside the archive.

    Parameters
    ----------
    chip_paths : List[str]
        Absolute paths of chip files to include, as returned by
        export_chips.

    Returns
    -------
    bytes
        Raw ZIP archive bytes with DEFLATE compression, ready to be passed
        directly to ``st.download_button(data=...)``.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in chip_paths:
            zf.write(path, arcname=os.path.basename(path))
    return buf.getvalue()
