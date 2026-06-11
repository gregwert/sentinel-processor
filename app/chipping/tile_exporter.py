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

import numpy as np
from PIL import Image


def _export_single_chip(args) -> str:
    """Top-level picklable worker: write one chip to disk in specified format."""
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
    """Build a filesystem-safe chip filename from the chip's geotransform origin."""
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


def export_chips(grid, output_dir: str, fmt: str = "png", naming: str = "rowcol", normalise: bool = False, global_stats=None):
    """Export all chips via ProcessPoolExecutor; return list of written paths."""
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


def zip_export(chip_paths) -> bytes:
    """Return in-memory ZIP bytes of chip files for st.download_button."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in chip_paths:
            zf.write(path, arcname=os.path.basename(path))
    return buf.getvalue()
