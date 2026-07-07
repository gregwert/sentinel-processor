"""Export endpoints: assemble and download chip ZIP archives with optional annotations."""
import asyncio
import json
import os
import tempfile
import zipfile

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from sentinel_backend.models import ExportRequest, ExportResult
from sentinel_backend.jobs import create_job, run_job
from sentinel_backend.storage import SessionStorage
from sentinel_backend.chipping.gdal_chipper import ChipGrid
from sentinel_backend.chipping.tile_exporter import ExportConfig, export_chips
from sentinel_backend.chipping.manifest import build_manifest, write_manifest_csv
from sentinel_backend.chipping.annotation_export import build_coco_manifest, build_yolo_files
from sentinel_backend.api.deps import validate_uuid, get_session_with_grid as _resolve_session_grid


_background_tasks: set = set()

router = APIRouter()


# ---------------------------------------------------------------------------
# Helper: rebuild ChipGrid from session artifacts
# ---------------------------------------------------------------------------

def _rebuild_grid(s: SessionStorage) -> ChipGrid:
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
# Export job worker
# ---------------------------------------------------------------------------

def _do_export(session_id: str, job_id: str, req: dict) -> dict:
    s = SessionStorage.get(session_id)

    if not s.grid_exists():
        raise RuntimeError("Chip grid not computed. Run PUT /chip-grid first.")

    grid = _rebuild_grid(s)

    # Compute global stats for optional z-score normalisation
    if s.artifact_exists("enhanced"):
        img = s.load_array("enhanced")
    elif s.artifact_exists("dehazed"):
        img = s.load_array("dehazed")
    else:
        img = s.load_array("stretched")

    arr_f = img.astype(np.float32)
    global_stats = {
        "mean": [float(arr_f[:, :, c].mean()) for c in range(3)],
        "std":  [float(arr_f[:, :, c].std())  for c in range(3)],
    }

    ref_stats = s.load_reference_stats() if s.reference_stats_exist() else None

    # Load filter results for rejected chip tracking
    rejected_indices = None
    if s.chip_stats_exist():
        filter_data = s.load_chip_stats()
        rejected_indices = filter_data.get("rejected", [])

    export_config = ExportConfig(
        fmt=req["chips_format"],
        naming=req["chips_naming"],
        normalise=req["chips_zscore_normalise"],
        global_stats=global_stats if req["chips_zscore_normalise"] else None,
        apply_ref_norm=req["chips_ref_norm_enabled"],
        reference_stats=ref_stats,
        ref_norm_method=req["chips_ref_norm_method"],
    )

    fmt_ext_map = {"png": ".png", "jpeg": ".jpg", "geotiff": ".tif", "npy": ".npy"}
    fmt_ext = fmt_ext_map.get(req["chips_format"], ".png")

    chip_stats = None
    if s.chip_stats_exist():
        chip_stats = s.load_chip_stats().get("stats")

    zip_path = str(s.export_path(job_id))
    n_files = 0

    with tempfile.TemporaryDirectory() as tmpdir:
        chips_dir = os.path.join(tmpdir, "chips")

        chip_paths = export_chips(
            grid,
            chips_dir,
            config=export_config,
            rejected_indices=rejected_indices,
            include_rejected=req["chips_include_rejected"],
        )

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            # Chips
            for path in chip_paths:
                zf.write(path, arcname=f"chips/{os.path.basename(path)}")
                n_files += 1

            # Manifest CSV
            rows = build_manifest(
                grid,
                chip_stats=chip_stats,
                naming=req["chips_naming"],
                fmt_ext=fmt_ext,
            )
            csv_bytes = write_manifest_csv(rows)
            zf.writestr("manifest.csv", csv_bytes)
            n_files += 1

            # COCO annotations
            if req["annotations_coco"]:
                coco = build_coco_manifest(
                    grid,
                    chip_stats=chip_stats,
                    naming=req["chips_naming"],
                    fmt_ext=fmt_ext,
                    rejected_indices=rejected_indices if not req["chips_include_rejected"] else None,
                    include_rejected=req["chips_include_rejected"],
                )
                zf.writestr("annotations_coco.json", json.dumps(coco, indent=2))
                n_files += 1

            # YOLO annotations
            if req["annotations_yolo"]:
                yolo_files = build_yolo_files(
                    grid,
                    chip_stats=chip_stats,
                    naming=req["chips_naming"],
                    fmt_ext=fmt_ext,
                    rejected_indices=rejected_indices if not req["chips_include_rejected"] else None,
                    include_rejected=req["chips_include_rejected"],
                )
                for archive_path, content in yolo_files.items():
                    zf.writestr(archive_path, content)
                    n_files += 1

    size_bytes = os.path.getsize(zip_path)

    result = ExportResult(
        download_url=f"/v1/sessions/{session_id}/export/{job_id}/download",
        size_bytes=size_bytes,
        n_files=n_files,
    )
    return result.model_dump()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("/{session_id}/export")
async def create_export(session_id: str, body: ExportRequest) -> dict:
    """Start a background job that assembles the chip export ZIP."""
    s = _resolve_session_grid(session_id)

    job_id = create_job()
    task = asyncio.create_task(run_job(job_id, _do_export, session_id, job_id, body.model_dump()))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return {"job_id": job_id}


@router.get("/{session_id}/export/{job_id}/download")
async def download_export(session_id: str, job_id: str) -> FileResponse:
    """Stream the completed export ZIP to the client."""
    validate_uuid(session_id, "session_id")
    validate_uuid(job_id, "job_id")
    s = _resolve_session_grid(session_id)

    zip_path = s.export_path(job_id)
    if not zip_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Export {job_id} not found. The job may still be running.",
        )

    return FileResponse(
        path=str(zip_path),
        media_type="application/zip",
        filename=f"export_{job_id}.zip",
    )
