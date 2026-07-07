"""Image processing endpoints: stretch, cloud mask, dehaze, enhance, preview, and histogram."""
import asyncio
import io

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from sentinel_backend.models import (
    CloudMaskRequest,
    CloudMaskResult,
    DehazeRequest,
    DehazeResult,
    EnhanceRequest,
    EnhanceResult,
    GeoMeta,
    HistogramResult,
    StretchRequest,
    StretchResult,
)
from sentinel_backend.jobs import create_job, run_job
from sentinel_backend.storage import SessionStorage
from sentinel_backend.api.deps import get_session as _resolve_session


_VALID_STAGES = {"stretched", "dehazed", "enhanced", "cloud_mask"}

_background_tasks: set = set()

router = APIRouter()


# ---------------------------------------------------------------------------
# Helper: load best available image for a session
# ---------------------------------------------------------------------------

def _best_image(s: SessionStorage) -> np.ndarray:
    """Return the most-processed array that exists in the session."""
    for stage in ("enhanced", "dehazed", "stretched"):
        if s.artifact_exists(stage):
            return s.load_array(stage)
    raise FileNotFoundError("No processed image found in session. Run /stretch first.")


# ---------------------------------------------------------------------------
# Stretch
# ---------------------------------------------------------------------------

def _do_stretch(session_id: str, req: dict) -> dict:
    from sentinel_backend.processing.preprocess import read_sentinel_tiff, percentile_stretch
    from sentinel_backend.processing.reference_norm import compute_reference_stats

    s = SessionStorage.get(session_id)
    tiff_path = str(s.source_tiff_path())

    arr, meta = read_sentinel_tiff(tiff_path, tuple(req["band_indices"]))
    stretched = percentile_stretch(arr, req["p_low"], req["p_high"], req["per_band"])
    s.save_array("stretched", stretched)

    geo_meta = GeoMeta.from_rasterio_meta(meta)
    s.save_meta(geo_meta)

    # Compute reference stats if refs exist
    ref_dir = s.ref_tiff_dir()
    ref_stats_summary = None
    ref_images = []
    for tif in ref_dir.glob("*.tif"):
        ref_arr, _ = read_sentinel_tiff(str(tif), tuple(req["band_indices"]))
        ref_stretched = percentile_stretch(ref_arr, req["p_low"], req["p_high"], req["per_band"])
        ref_images.append(ref_stretched)
    if ref_images:
        stats = compute_reference_stats(ref_images)
        s.save_reference_stats(stats)
        ref_stats_summary = {
            "n": stats["n"],
            "mean": stats["mean"].tolist(),
            "std": stats["std"].tolist(),
        }

    arr_f = stretched.astype(np.float32)
    stats_dict = {
        "mean": [float(arr_f[:, :, c].mean()) for c in range(arr_f.shape[2])],
        "std":  [float(arr_f[:, :, c].std())  for c in range(arr_f.shape[2])],
    }

    result = StretchResult(
        artifact="stretched",
        meta=geo_meta,
        stats=stats_dict,
        reference_stats_summary=ref_stats_summary,
    )
    return result.model_dump()


@router.post("/{session_id}/stretch")
async def stretch(session_id: str, body: StretchRequest) -> dict:
    """Start a background job that reads and percentile-stretches the source TIFF."""
    s = _resolve_session(session_id)

    job_id = create_job()
    task = asyncio.create_task(run_job(job_id, _do_stretch, session_id, body.model_dump()))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return {"job_id": job_id}


# ---------------------------------------------------------------------------
# Cloud mask (synchronous)
# ---------------------------------------------------------------------------

@router.post("/{session_id}/cloud-mask", response_model=CloudMaskResult)
async def cloud_mask(session_id: str, body: CloudMaskRequest) -> CloudMaskResult:
    """Detect clouds in the specified stage image and save a binary mask."""
    from sentinel_backend.processing.dehazing import detect_clouds_simple

    s = _resolve_session(session_id)

    if body.source_stage not in {"stretched", "dehazed", "enhanced"}:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid source_stage '{body.source_stage}'. Must be one of: {sorted({'stretched', 'dehazed', 'enhanced'})}",
        )

    try:
        img = s.load_array(body.source_stage)
    except FileNotFoundError:
        raise HTTPException(
            status_code=422,
            detail=f"Stage '{body.source_stage}' not found. Run /stretch first.",
        )

    mask = detect_clouds_simple(img, body.brightness_thresh, body.saturation_thresh)
    s.save_array("cloud_mask", mask.astype(np.uint8))

    cloud_px = int(mask.sum())
    total_px = int(mask.size)
    return CloudMaskResult(
        cloud_pct=cloud_px / total_px,
        cloud_px=cloud_px,
        total_px=total_px,
    )


# ---------------------------------------------------------------------------
# Dehaze
# ---------------------------------------------------------------------------

def _do_dehaze(session_id: str, req: dict) -> dict:
    from sentinel_backend.processing.dehazing import Dehazer

    s = SessionStorage.get(session_id)
    img = s.load_array("stretched")

    dehazer = Dehazer(
        patch_size=req["patch_size"],
        omega=req["omega"],
        t0=req["t0"],
        use_guided_filter=req["use_guided_filter"],
        mask_clouds=req["mask_clouds"],
        brightness_thresh=req["brightness_thresh"],
        saturation_thresh=req["saturation_thresh"],
    )
    dehazed = dehazer.run(img)
    s.save_array("dehazed", dehazed)

    arr_f = dehazed.astype(np.float32)
    result = DehazeResult(
        artifact="dehazed",
        stats={"mean": float(arr_f.mean()), "std": float(arr_f.std())},
    )
    return result.model_dump()


@router.post("/{session_id}/dehaze")
async def dehaze(session_id: str, body: DehazeRequest) -> dict:
    """Start a background job that applies DCP dehazing to the stretched image."""
    s = _resolve_session(session_id)

    if not s.artifact_exists("stretched"):
        raise HTTPException(status_code=422, detail="Run /stretch before /dehaze.")

    job_id = create_job()
    task = asyncio.create_task(run_job(job_id, _do_dehaze, session_id, body.model_dump()))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return {"job_id": job_id}


# ---------------------------------------------------------------------------
# Enhance
# ---------------------------------------------------------------------------

def _do_enhance(session_id: str, req: dict) -> dict:
    from sentinel_backend.processing.enhancement import apply_gray_world, apply_clahe
    from sentinel_backend.processing.reference_norm import apply_reference_normalisation

    s = SessionStorage.get(session_id)

    # Use most-processed available source
    if s.artifact_exists("dehazed"):
        img = s.load_array("dehazed")
    else:
        img = s.load_array("stretched")

    # Gray World white balance
    if req["gray_world"]:
        cloud_mask = None
        if s.artifact_exists("cloud_mask"):
            cloud_mask = s.load_array("cloud_mask").astype(bool)
        img = apply_gray_world(img, cloud_mask)

    # Reference normalisation
    if req["ref_norm_enabled"]:
        ref_stats = s.load_reference_stats()
        if ref_stats is not None:
            img = apply_reference_normalisation(img, ref_stats, req["ref_norm_method"])

    # CLAHE
    if req["clahe_enabled"]:
        img = apply_clahe(img, req["clahe_clip_limit"], tuple(req["clahe_tile_grid"]))

    s.save_array("enhanced", img)

    arr_f = img.astype(np.float32)
    global_stats = {
        "mean": [float(arr_f[:, :, c].mean()) for c in range(arr_f.shape[2])],
        "std":  [float(arr_f[:, :, c].std())  for c in range(arr_f.shape[2])],
    }

    result = EnhanceResult(artifact="enhanced", global_stats=global_stats)
    return result.model_dump()


@router.post("/{session_id}/enhance")
async def enhance(session_id: str, body: EnhanceRequest) -> dict:
    """Start a background job that applies enhancement (gray world / ref norm / CLAHE)."""
    s = _resolve_session(session_id)

    if not s.artifact_exists("stretched") and not s.artifact_exists("dehazed"):
        raise HTTPException(status_code=422, detail="Run /stretch (and optionally /dehaze) before /enhance.")

    job_id = create_job()
    task = asyncio.create_task(run_job(job_id, _do_enhance, session_id, body.model_dump()))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return {"job_id": job_id}


# ---------------------------------------------------------------------------
# Preview
# ---------------------------------------------------------------------------

@router.get("/{session_id}/preview/{stage}")
async def preview(session_id: str, stage: str, max_px: int = 900) -> Response:
    """Return a PNG preview of a processing stage, downsampled to max_px on the long edge."""
    from PIL import Image

    s = _resolve_session(session_id)

    if stage not in _VALID_STAGES:
        raise HTTPException(status_code=400, detail=f"Invalid stage '{stage}'. Must be one of: {sorted(_VALID_STAGES)}")

    try:
        img_arr = s.load_array(stage)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Stage '{stage}' not found in session.")

    pil_img = Image.fromarray(img_arr)
    h, w = img_arr.shape[:2]
    scale = max_px / max(h, w)
    if scale < 1.0:
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)

    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


# ---------------------------------------------------------------------------
# Histogram
# ---------------------------------------------------------------------------

@router.get("/{session_id}/histogram", response_model=HistogramResult)
async def histogram(
    session_id: str,
    stage: str = "stretched",
    n: int = 200000,
) -> HistogramResult:
    """Return an 80-bin brightness histogram split by cloud / land label."""
    s = _resolve_session(session_id)

    if stage not in _VALID_STAGES:
        raise HTTPException(status_code=400, detail=f"Invalid stage '{stage}'. Must be one of: {sorted(_VALID_STAGES)}")

    try:
        img_arr = s.load_array(stage)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Stage '{stage}' not found in session.")

    cloud_mask = None
    if s.artifact_exists("cloud_mask"):
        cloud_mask = s.load_array("cloud_mask").astype(bool)

    # Compute per-pixel mean brightness
    brightness = img_arr.astype(np.float32).mean(axis=2).ravel()
    mask_flat = cloud_mask.ravel() if cloud_mask is not None else None

    # Subsample to at most n pixels for speed
    total = len(brightness)
    if total > n:
        rng = np.random.default_rng(42)
        idx = rng.choice(total, n, replace=False)
        brightness = brightness[idx]
        if mask_flat is not None:
            mask_flat = mask_flat[idx]

    bins = np.linspace(0.0, 255.0, 81)  # 80 bins → 81 edges

    if mask_flat is not None:
        cloud_counts, _ = np.histogram(brightness[mask_flat],  bins=bins)
        land_counts,  _ = np.histogram(brightness[~mask_flat], bins=bins)
    else:
        land_counts,  _ = np.histogram(brightness, bins=bins)
        cloud_counts = np.zeros(80, dtype=np.int64)

    return HistogramResult(
        bins=bins.tolist(),
        cloud_counts=cloud_counts.tolist(),
        land_counts=land_counts.tolist(),
    )
