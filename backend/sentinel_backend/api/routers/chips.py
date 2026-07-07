"""Chip grid, filter, list, thumbnail, and manifest CSV endpoints."""
import io

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from sentinel_backend.models import (
    ChipFiltersRequest,
    ChipFiltersResult,
    ChipGridRequest,
    ChipGridSpec,
    ChipItem,
    ChipListResult,
)
from sentinel_backend.storage import SessionStorage
from sentinel_backend.chipping.gdal_chipper import ChipGrid, build_chip_grid, get_chip
from sentinel_backend.chipping.chip_filter import apply_chip_filters
from sentinel_backend.chipping.manifest import build_manifest, write_manifest_csv
from sentinel_backend.api.deps import get_session as _resolve_session, get_session_with_grid as _resolve_session_grid


router = APIRouter()


# ---------------------------------------------------------------------------
# Helper: reconstruct ChipGrid from a saved spec + loaded image
# ---------------------------------------------------------------------------

def _rebuild_grid(s: SessionStorage) -> ChipGrid:
    """Reconstruct a ChipGrid from saved spec and best available image."""
    spec = s.load_grid()
    meta = s.load_meta()
    source_meta = meta.to_rasterio_meta()

    if s.artifact_exists("enhanced"):
        img = s.load_array("enhanced")
    elif s.artifact_exists("dehazed"):
        img = s.load_array("dehazed")
    else:
        img = s.load_array("stretched")

    return ChipGrid(
        windows=[tuple(w) for w in spec.windows],
        source_image=img,
        source_meta=source_meta,
        chip_w=spec.chip_w,
        chip_h=spec.chip_h,
        n_rows=spec.n_rows,
        n_cols=spec.n_cols,
        edge_mode=spec.edge_mode,
    )


# ---------------------------------------------------------------------------
# Chip grid
# ---------------------------------------------------------------------------

@router.put("/{session_id}/chip-grid", response_model=ChipGridSpec)
async def set_chip_grid(session_id: str, body: ChipGridRequest) -> ChipGridSpec:
    """Compute and persist a chip grid layout for the current enhanced image."""
    s = _resolve_session(session_id)

    # Load best available image
    if s.artifact_exists("enhanced"):
        img = s.load_array("enhanced")
    elif s.artifact_exists("dehazed"):
        img = s.load_array("dehazed")
    elif s.artifact_exists("stretched"):
        img = s.load_array("stretched")
    else:
        raise HTTPException(status_code=422, detail="No processed image found. Run /stretch first.")

    try:
        meta = s.load_meta()
    except Exception:
        raise HTTPException(status_code=422, detail="Session metadata not found. Run /stretch first.")

    source_meta = meta.to_rasterio_meta()
    grid = build_chip_grid(img, source_meta, body.chip_w, body.chip_h, body.overlap, body.edge_mode)

    spec = ChipGridSpec(
        chip_w=grid.chip_w,
        chip_h=grid.chip_h,
        overlap=body.overlap,
        edge_mode=grid.edge_mode,
        naming=body.naming,
        n_rows=grid.n_rows,
        n_cols=grid.n_cols,
        total=grid.total,
        windows=[[c, r, w, h] for c, r, w, h in grid.windows],
    )
    s.save_grid(spec)
    return spec


# ---------------------------------------------------------------------------
# Chip filters
# ---------------------------------------------------------------------------

@router.post("/{session_id}/chip-filters", response_model=ChipFiltersResult)
async def chip_filters(session_id: str, body: ChipFiltersRequest) -> ChipFiltersResult:
    """Apply cloud-coverage and variance filters across all chips."""
    s = _resolve_session_grid(session_id)

    grid = _rebuild_grid(s)

    cloud_mask = None
    if s.artifact_exists("cloud_mask"):
        cloud_mask = s.load_array("cloud_mask").astype(bool)

    accepted, rejected, stats = apply_chip_filters(
        grid,
        cloud_mask,
        cloud_thresh=body.cloud_thresh,
        variance_thresh=body.variance_thresh,
        enable_cloud_filter=body.cloud_enabled,
        enable_variance_filter=body.variance_enabled,
    )

    # Persist filter results so the chips list and manifest can use them
    s.save_chip_stats(accepted, rejected, stats)

    return ChipFiltersResult(accepted=accepted, rejected=rejected, stats=stats)


# ---------------------------------------------------------------------------
# Chips list
# ---------------------------------------------------------------------------

@router.get("/{session_id}/chips", response_model=ChipListResult)
async def list_chips(
    session_id: str,
    page: int = 0,
    page_size: int = 16,
    include_rejected: bool = False,
) -> ChipListResult:
    """Return a paginated list of chips with thumbnail URLs."""
    s = _resolve_session_grid(session_id)

    spec = s.load_grid()

    # Load persisted filter results if available
    rejected_set: set[int] = set()
    if s.chip_stats_exist():
        filter_data = s.load_chip_stats()
        rejected_set = set(filter_data.get("rejected", []))

    # Build index list respecting include_rejected
    if include_rejected or not rejected_set:
        all_indices = list(range(spec.total))
    else:
        all_indices = [i for i in range(spec.total) if i not in rejected_set]

    total = len(all_indices)
    start = page * page_size
    page_indices = all_indices[start: start + page_size]

    items = []
    for idx in page_indices:
        row_idx = idx // spec.n_cols
        col_idx = idx % spec.n_cols
        items.append(ChipItem(
            index=idx,
            row=row_idx,
            col=col_idx,
            thumbnail_url=f"/v1/sessions/{session_id}/chips/{idx}/thumbnail.png",
            rejected=idx in rejected_set,
        ))

    return ChipListResult(total=total, page=page, page_size=page_size, items=items)


# ---------------------------------------------------------------------------
# Chip thumbnail
# ---------------------------------------------------------------------------

@router.get("/{session_id}/chips/{index}/thumbnail.png")
async def chip_thumbnail(session_id: str, index: int) -> Response:
    """Return a 128×128 PNG thumbnail for the chip at the given flat index."""
    from PIL import Image

    s = _resolve_session_grid(session_id)

    grid = _rebuild_grid(s)

    if index < 0 or index >= grid.total:
        raise HTTPException(status_code=404, detail=f"Chip index {index} out of range (total={grid.total}).")

    chip_arr, _ = get_chip(grid, index)
    pil_img = Image.fromarray(chip_arr).resize((128, 128), Image.LANCZOS)

    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


# ---------------------------------------------------------------------------
# Manifest CSV
# ---------------------------------------------------------------------------

@router.get("/{session_id}/manifest.csv")
async def get_manifest_csv(
    session_id: str,
    naming: str = "coords",
    fmt_ext: str = ".png",
    include_rejected: bool = False,
) -> Response:
    """Return a CSV manifest with per-chip bounds and quality stats."""
    s = _resolve_session_grid(session_id)

    grid = _rebuild_grid(s)

    # Load persisted chip stats if available
    chip_stats = None
    if s.chip_stats_exist():
        filter_data = s.load_chip_stats()
        chip_stats = filter_data.get("stats")

    rows = build_manifest(grid, chip_stats=chip_stats, naming=naming, fmt_ext=fmt_ext)

    if not include_rejected and chip_stats:
        rejected_set = {st["chip_index"] for st in chip_stats if st.get("rejected")}
        rows = [r for r in rows if r["chip_index"] not in rejected_set]

    csv_bytes = write_manifest_csv(rows)
    return Response(
        content=csv_bytes,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=manifest.csv"},
    )
