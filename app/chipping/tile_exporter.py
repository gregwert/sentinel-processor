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

    Attributes:
        fmt (str): Output format. One of 'png', 'jpeg', 'geotiff', 'npy'.
        naming (str): Chip filename scheme. 'rowcol' produces r0001_c0002; 'coords' derives
            names from lat/lon or easting/northing.
        normalise (bool): If True, apply z-score normalisation using global_stats before
            export.
        global_stats (dict or None): Must be provided when normalise=True. Dict with keys
            'mean' and 'std', each a list of 3 floats (one per channel).
        apply_ref_norm (bool): When True, apply reference-based radiometric normalisation to
            each chip before format conversion. reference_stats must be provided.
        reference_stats (dict or None): Pre-computed reference statistics as returned by
            processing.reference_norm.compute_reference_stats. Required when apply_ref_norm
            is True.
        ref_norm_method (str): One of 'histogram' or 'linear'. Passed to
            apply_reference_normalisation. Default 'histogram'.
    """

    fmt: str = "png"                  # output file format: 'png', 'jpeg', 'geotiff', or 'npy'
    naming: str = "coords"            # filename scheme: 'rowcol' or 'coords'
    normalise: bool = False           # whether to apply z-score normalisation before export
    global_stats: dict = None         # dict with 'mean' and 'std' lists (3 floats each) for normalisation
    apply_ref_norm: bool = False
    reference_stats: dict = None
    ref_norm_method: str = "histogram"


@dataclass
class _ChipTask:
    """Single-chip export task passed to the process pool worker.

    Groups per-chip data with the shared ExportConfig so the worker
    receives one object instead of a positional tuple.
    """
    chip_array: np.ndarray
    chip_meta: dict
    out_path: str
    config: ExportConfig


def _export_single_chip(task: "_ChipTask") -> str:
    """Write one chip to disk; used as a top-level picklable worker.

    Receives a _ChipTask so it can be passed to executor.map without a
    lambda (lambdas are not picklable).

    Args:
        task: Contains the chip array, metadata, output path, and the
            shared ExportConfig for this export run.

    Returns:
        The out_path that was written.
    """
    chip_array = task.chip_array
    chip_meta = task.chip_meta
    out_path = task.out_path
    cfg = task.config

    fmt = cfg.fmt
    global_mean = cfg.global_stats.get("mean") if cfg.global_stats else None
    global_std = cfg.global_stats.get("std") if cfg.global_stats else None

    if cfg.normalise and global_mean is not None and global_std is not None:
        mean = np.array(global_mean, dtype=np.float32)
        std = np.array(global_std, dtype=np.float32) + 1e-6
        chip_f = chip_array.astype(np.float32)
        normalised = (chip_f - mean) / std
        if fmt == "npy":
            np.save(out_path, normalised)
            return out_path
        else:
            rescaled = normalised * 45.0 + 127.5
            chip_array = np.clip(rescaled, 0, 255).astype(np.uint8)

    if cfg.apply_ref_norm and cfg.reference_stats is not None:
        from processing.reference_norm import apply_reference_normalisation
        chip_array = apply_reference_normalisation(chip_array, cfg.reference_stats, cfg.ref_norm_method)

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

    Args:
        chip_meta (dict): Chip metadata dict as returned by get_chip. Must contain a
            'transform' key (rasterio Affine) and an optional 'crs' key (rasterio CRS).
        ext (str): File extension including the leading dot, e.g. '.png' or '.tif'.

    Returns:
        str: Filename string such as 'chip_51p4823N_000p1034W.png' (geographic) or
            'chip_5714000N_341000E.tif' (projected).
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
                 config: "ExportConfig | None" = None,
                 rejected_indices: "list[int] | None" = None,
                 include_rejected: bool = False) -> List[str]:
    """Export all chips in a ChipGrid to disk using a process pool.

    Individual keyword arguments (``fmt``, ``naming``, ``normalise``,
    ``global_stats``) are provided for backward compatibility.  When
    ``config`` is given it takes precedence over those arguments, so
    existing call sites that pass individual kwargs continue to work
    without any changes.

    Args:
        grid (ChipGrid): The chip grid produced by build_chip_grid. Provides window list,
            source image reference, and spatial metadata.
        output_dir (str): Directory to write chip files into. Created if it does not exist.
        fmt (str, optional): Output format. One of 'png', 'jpeg', 'geotiff', 'npy'.
            Default is 'png'.
        naming (str, optional): Chip filename scheme. 'rowcol' produces names like
            'chip_r0001_c0002.png'; 'coords' derives names from the chip's geographic
            origin via _coords_filename. Default is 'rowcol'.
        normalise (bool, optional): If True, apply z-score normalisation using global_stats
            before writing. Default is False.
        global_stats (dict or None, optional): Required when normalise=True. Dict with keys
            'mean' and 'std', each a list of 3 floats (one per channel). Default is None.
        config (ExportConfig or None, optional): If provided, its values override fmt,
            naming, normalise, and global_stats. Default is None.
        rejected_indices (list[int] or None, optional): Chip indices that failed quality
            filters. When provided and ``include_rejected`` is False, these chips are
            skipped during export. Default is None (no chips skipped).
        include_rejected (bool, optional): When True, rejected chips are included in the
            export even if ``rejected_indices`` is provided. Default is False.

    Returns:
        List[str]: Absolute paths of all written chip files, in window order (row-major,
            matching grid.windows).
    """
    if config is None:
        config = ExportConfig(fmt=fmt, naming=naming, normalise=normalise, global_stats=global_stats)
    os.makedirs(output_dir, exist_ok=True)

    ext_map = {
        "png": ".png",
        "jpeg": ".jpg",
        "geotiff": ".tif",
        "npy": ".npy",
    }
    ext = ext_map.get(config.fmt, ".png")

    from chipping.gdal_chipper import get_chip

    skip_set = set(rejected_indices) if (rejected_indices and not include_rejected) else set()

    tasks = []
    for index in range(grid.total):
        if index in skip_set:
            continue

        chip_array, chip_meta = get_chip(grid, index)
        row_idx = chip_meta["row_idx"]
        col_idx = chip_meta["col_idx"]

        if config.naming == "rowcol":
            fname = f"chip_r{row_idx:04d}_c{col_idx:04d}{ext}"
        elif config.naming == "coords":
            fname = _coords_filename(chip_meta, ext)
        else:
            fname = f"chip_r{row_idx:04d}_c{col_idx:04d}{ext}"

        out_path = os.path.join(output_dir, fname)
        tasks.append(_ChipTask(chip_array=chip_array, chip_meta=chip_meta, out_path=out_path, config=config))

    workers = os.cpu_count() or 4
    ctx = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as executor:
        results = list(executor.map(_export_single_chip, tasks))

    return results


def zip_export(chip_paths: List[str]) -> bytes:
    """Bundle a list of chip files into an in-memory ZIP archive.

    Intended for use with Streamlit's ``st.download_button`` so that a full
    chip set can be offered as a single download without writing a ZIP to
    disk.  Each chip is stored under its bare filename (no directory path)
    inside the archive.

    Args:
        chip_paths (List[str]): Absolute paths of chip files to include, as returned by
            export_chips.

    Returns:
        bytes: Raw ZIP archive bytes with DEFLATE compression, ready to be passed directly
            to ``st.download_button(data=...)``.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in chip_paths:
            zf.write(path, arcname=os.path.basename(path))
    return buf.getvalue()
