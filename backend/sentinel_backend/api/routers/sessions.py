"""Session lifecycle endpoints: create, retrieve, delete, and upload source/reference GeoTIFFs."""
from datetime import datetime, timezone
from pathlib import Path
from fastapi import APIRouter, File, HTTPException, UploadFile

import rasterio

from sentinel_backend.models import SessionInfo
from sentinel_backend.storage import SessionStorage
from sentinel_backend.api.deps import validate_uuid, get_session as _resolve_session


router = APIRouter()

CHUNK_SIZE = 1 << 20  # 1 MB chunks for streaming uploads


def _session_info(s: SessionStorage) -> SessionInfo:
    info = s.info()
    return SessionInfo(
        session_id=s.session_id,
        created_at=datetime.fromtimestamp(info["created_at"], tz=timezone.utc).isoformat(),
        expires_at=datetime.fromtimestamp(info["expires_at"], tz=timezone.utc).isoformat(),
        stages_ready=s.stages_ready(),
    )


@router.post("", status_code=201, response_model=SessionInfo)
async def create_session() -> SessionInfo:
    """Create a new processing session and return its metadata."""
    s = SessionStorage.create()
    return _session_info(s)


@router.get("/{session_id}", response_model=SessionInfo)
async def get_session(session_id: str) -> SessionInfo:
    """Return metadata for an existing session."""
    s = _resolve_session(session_id)
    return _session_info(s)


@router.delete("/{session_id}", status_code=204)
async def delete_session(session_id: str):
    """Delete a session and all its artifacts."""
    validate_uuid(session_id, "session_id")
    SessionStorage.delete(session_id)


@router.post("/{session_id}/source")
async def upload_source(session_id: str, file: UploadFile = File(...)) -> dict:
    """Stream a source GeoTIFF to the session workspace and return basic raster info."""
    s = _resolve_session(session_id)

    tiff_path = s.source_tiff_path()
    with open(tiff_path, "wb") as f:
        while True:
            chunk = await file.read(CHUNK_SIZE)
            if not chunk:
                break
            f.write(chunk)

    try:
        with rasterio.open(str(tiff_path)) as src:
            return {
                "width": src.width,
                "height": src.height,
                "band_count": src.count,
                "filename": file.filename,
            }
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not open uploaded file as raster: {exc}")


@router.post("/{session_id}/references")
async def upload_references(session_id: str, files: list[UploadFile] = File(...)) -> dict:
    """Upload one or more reference GeoTIFFs to the session refs/ directory."""
    s = _resolve_session(session_id)

    ref_dir = s.ref_tiff_dir()
    saved = []
    for upload in files:
        dest = ref_dir / (Path(upload.filename).name if upload.filename else f"ref_{len(saved)}.tif")
        with open(dest, "wb") as f:
            while True:
                chunk = await upload.read(CHUNK_SIZE)
                if not chunk:
                    break
                f.write(chunk)
        saved.append(upload.filename)

    return {"saved": saved}
