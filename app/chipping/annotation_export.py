"""COCO JSON and YOLO annotation export for unannotated satellite chip datasets."""
import json
import io
from datetime import date


def build_coco_manifest(
    grid,
    chip_stats: list | None = None,
    naming: str = "rowcol",
    fmt_ext: str = ".png",
    rejected_indices: list | None = None,
    include_rejected: bool = False,
) -> dict:
    """Build a COCO-format dataset manifest dict for a chip grid.

    Chips are exported as unannotated images. Geographic bounding boxes
    are stored in a non-standard ``geo_bbox`` field on each image entry.

    Args:
        grid (ChipGrid): The chip grid to build the manifest for.
        chip_stats (list[dict] or None): Per-chip stats from apply_chip_filters (keys:
            chip_index, cloud_pct, variance, rejected). When None, geo_bbox is omitted.
        naming (str): 'rowcol' or 'coords'.
        fmt_ext (str): File extension including dot, e.g. '.png'.
        rejected_indices (list[int] or None): Chip indices to exclude when
            include_rejected is False.
        include_rejected (bool): When False, rejected chips are omitted from the manifest.

    Returns:
        dict: COCO-format manifest with keys: info, images, annotations, categories.
    """
    from chipping.manifest import chip_lat_lon_bounds
    from chipping.tile_exporter import _coords_filename
    from chipping.gdal_chipper import get_chip

    skip_set = set(rejected_indices) if (rejected_indices and not include_rejected) else set()

    transform = grid.source_meta.get("transform")
    crs = grid.source_meta.get("crs")
    n_cols = grid.n_cols

    images = []
    for idx, (col_off, row_off, chip_w, chip_h) in enumerate(grid.windows):
        if idx in skip_set:
            continue

        row_idx = idx // n_cols
        col_idx = idx % n_cols

        if naming == "coords" and transform is not None and crs is not None:
            _, chip_meta = get_chip(grid, idx)
            filename = _coords_filename(chip_meta, fmt_ext)
        else:
            filename = f"chip_r{row_idx:04d}_c{col_idx:04d}{fmt_ext}"

        entry = {
            "id": idx + 1,  # COCO ids are 1-based
            "file_name": f"chips/{filename}",
            "width": chip_w,
            "height": chip_h,
        }

        lon_min, lat_min, lon_max, lat_max = chip_lat_lon_bounds(
            transform, crs, col_off, row_off, chip_w, chip_h
        )
        if lon_min is not None:
            entry["geo_bbox"] = [
                round(lon_min, 6), round(lat_min, 6),
                round(lon_max, 6), round(lat_max, 6),
            ]

        images.append(entry)

    return {
        "info": {
            "description": "Sentinel-2 chip dataset — unannotated",
            "source": "sentinel-processor",
            "date_created": date.today().isoformat(),
            "note": "geo_bbox is a non-standard extension: [lon_min, lat_min, lon_max, lat_max] in WGS-84.",
        },
        "images": images,
        "annotations": [],
        "categories": [],
    }


def build_yolo_files(
    grid,
    chip_stats: list | None = None,
    naming: str = "rowcol",
    fmt_ext: str = ".png",
    rejected_indices: list | None = None,
    include_rejected: bool = False,
) -> dict:
    """Build per-chip YOLO annotation .txt files and a dataset.yaml.

    Each chip gets an empty .txt file (no objects annotated). A dataset.yaml
    describing the structure is also produced.

    Args:
        grid (ChipGrid): The chip grid to build annotation files for.
        chip_stats (list[dict] or None): Not used for content, but accepted for API
            consistency.
        naming (str): 'rowcol' or 'coords'.
        fmt_ext (str): File extension including dot, e.g. '.png'.
        rejected_indices (list[int] or None): Chip indices to exclude when
            include_rejected is False.
        include_rejected (bool): When False, rejected chips are omitted.

    Returns:
        dict[str, bytes]: Mapping of archive path to file bytes.
            Keys: ``"labels/<filename>.txt"`` for each chip, ``"dataset.yaml"``.
    """
    from chipping.tile_exporter import _coords_filename
    from chipping.gdal_chipper import get_chip

    skip_set = set(rejected_indices) if (rejected_indices and not include_rejected) else set()

    transform = grid.source_meta.get("transform")
    crs = grid.source_meta.get("crs")
    n_cols = grid.n_cols

    files = {}
    chip_filenames = []

    for idx, (col_off, row_off, chip_w, chip_h) in enumerate(grid.windows):
        if idx in skip_set:
            continue

        row_idx = idx // n_cols
        col_idx = idx % n_cols

        if naming == "coords" and transform is not None and crs is not None:
            _, chip_meta = get_chip(grid, idx)
            chip_filename = _coords_filename(chip_meta, fmt_ext)
        else:
            chip_filename = f"chip_r{row_idx:04d}_c{col_idx:04d}{fmt_ext}"

        chip_filenames.append(chip_filename)

        # Stem for the label file (strip extension)
        stem = chip_filename.rsplit(".", 1)[0]
        files[f"labels/{stem}.txt"] = b""  # empty = no annotations

    files["dataset.yaml"] = _build_dataset_yaml(chip_filenames)
    return files


def _build_dataset_yaml(chip_filenames: list) -> bytes:
    """Build a minimal YOLO dataset.yaml as UTF-8 bytes.

    Image files are assumed to be under chips/ relative to the ZIP root.
    Label files are under labels/.
    """
    lines = [
        "# Unannotated satellite chip dataset exported by sentinel-processor.",
        "# Add object annotations to the labels/ .txt files before training.",
        "# Each chip image is in chips/, each label file in labels/ with matching stem.",
        "",
        f"path: .",
        f"train: chips",
        f"val: chips",
        f"nc: 0",
        f"names: []",
        "",
        f"# Total chips: {len(chip_filenames)}",
    ]
    return "\n".join(lines).encode("utf-8")
