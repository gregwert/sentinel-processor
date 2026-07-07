"""
api_client.py — Thin synchronous httpx wrapper around the sentinel-processor backend API.

All calls are synchronous (httpx sync client) because Streamlit runs synchronously.
Set the BACKEND_URL environment variable to point at the backend service
(default: http://localhost:8000).
"""
import os
import time
import httpx
import streamlit as st
from sentinel_frontend.models import JobRecord, JobStatus

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")
_TIMEOUT = httpx.Timeout(300.0)  # 5 min for long jobs


def _client() -> httpx.Client:
    return httpx.Client(base_url=BACKEND_URL, timeout=_TIMEOUT)


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------

def create_session() -> str:
    with _client() as c:
        r = c.post("/v1/sessions")
        r.raise_for_status()
        return r.json()["session_id"]


def delete_session(session_id: str):
    with _client() as c:
        c.delete(f"/v1/sessions/{session_id}")


# ---------------------------------------------------------------------------
# Source / reference upload
# ---------------------------------------------------------------------------

def upload_source(session_id: str, file_bytes: bytes, filename: str) -> dict:
    with _client() as c:
        r = c.post(
            f"/v1/sessions/{session_id}/source",
            files={"file": (filename, file_bytes, "image/tiff")},
        )
        r.raise_for_status()
        return r.json()


def upload_references(session_id: str, files: list[tuple[str, bytes]]) -> dict:
    with _client() as c:
        r = c.post(
            f"/v1/sessions/{session_id}/references",
            files=[("files", (name, data, "image/tiff")) for name, data in files],
        )
        r.raise_for_status()
        return r.json()


# ---------------------------------------------------------------------------
# Processing jobs
# ---------------------------------------------------------------------------

def run_stretch(
    session_id: str,
    band_indices,
    p_low: float,
    p_high: float,
    per_band: bool,
) -> str:
    with _client() as c:
        r = c.post(
            f"/v1/sessions/{session_id}/stretch",
            json={
                "band_indices": list(band_indices),
                "p_low": p_low,
                "p_high": p_high,
                "per_band": per_band,
            },
        )
        r.raise_for_status()
        return r.json()["job_id"]


def update_cloud_mask(
    session_id: str,
    brightness_thresh: float,
    saturation_thresh: float,
    source_stage: str = "stretched",
) -> dict:
    with _client() as c:
        r = c.post(
            f"/v1/sessions/{session_id}/cloud-mask",
            json={
                "brightness_thresh": brightness_thresh,
                "saturation_thresh": saturation_thresh,
                "source_stage": source_stage,
            },
        )
        r.raise_for_status()
        return r.json()


def run_dehaze(session_id: str, params: dict) -> str:
    with _client() as c:
        r = c.post(f"/v1/sessions/{session_id}/dehaze", json=params)
        r.raise_for_status()
        return r.json()["job_id"]


def run_enhance(session_id: str, params: dict) -> str:
    with _client() as c:
        r = c.post(f"/v1/sessions/{session_id}/enhance", json=params)
        r.raise_for_status()
        return r.json()["job_id"]


# ---------------------------------------------------------------------------
# Preview / histogram
# ---------------------------------------------------------------------------

def fetch_preview(session_id: str, stage: str, max_px: int = 900) -> bytes:
    with _client() as c:
        r = c.get(
            f"/v1/sessions/{session_id}/preview/{stage}",
            params={"max_px": max_px},
        )
        r.raise_for_status()
        return r.content


def fetch_best_preview(session_id: str, max_px: int = 900) -> bytes:
    """Fetch the most-processed available stage, falling back through enhanced→dehazed→stretched."""
    for stage in ("enhanced", "dehazed", "stretched"):
        try:
            return fetch_preview(session_id, stage, max_px)
        except Exception:
            continue
    raise RuntimeError("No processed image available in session")


def fetch_thumbnail(session_id: str, index: int) -> bytes:
    with _client() as c:
        r = c.get(f"/v1/sessions/{session_id}/chips/{index}/thumbnail.png")
        r.raise_for_status()
        return r.content


def get_histogram(session_id: str, stage: str = "stretched", n: int = 200_000) -> dict:
    with _client() as c:
        r = c.get(
            f"/v1/sessions/{session_id}/histogram",
            params={"stage": stage, "n": n},
        )
        r.raise_for_status()
        return r.json()


# ---------------------------------------------------------------------------
# Chip grid
# ---------------------------------------------------------------------------

def put_chip_grid(
    session_id: str,
    chip_w: int,
    chip_h: int,
    overlap: float,
    edge_mode: str,
    naming: str,
) -> dict:
    with _client() as c:
        r = c.put(
            f"/v1/sessions/{session_id}/chip-grid",
            json={
                "chip_w": chip_w,
                "chip_h": chip_h,
                "overlap": overlap,
                "edge_mode": edge_mode,
                "naming": naming,
            },
        )
        r.raise_for_status()
        return r.json()


def run_chip_filters(
    session_id: str,
    cloud_enabled: bool,
    cloud_thresh: float,
    variance_enabled: bool,
    variance_thresh: float,
) -> dict:
    with _client() as c:
        r = c.post(
            f"/v1/sessions/{session_id}/chip-filters",
            json={
                "cloud_enabled": cloud_enabled,
                "cloud_thresh": cloud_thresh,
                "variance_enabled": variance_enabled,
                "variance_thresh": variance_thresh,
            },
        )
        r.raise_for_status()
        return r.json()


def list_chips(
    session_id: str,
    page: int = 0,
    page_size: int = 16,
    include_rejected: bool = False,
) -> dict:
    with _client() as c:
        r = c.get(
            f"/v1/sessions/{session_id}/chips",
            params={
                "page": page,
                "page_size": page_size,
                "include_rejected": include_rejected,
            },
        )
        r.raise_for_status()
        return r.json()


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def run_export(session_id: str, export_request: dict) -> str:
    with _client() as c:
        r = c.post(f"/v1/sessions/{session_id}/export", json=export_request)
        r.raise_for_status()
        return r.json()["job_id"]


def download_export(session_id: str, job_id: str) -> bytes:
    with _client() as c:
        r = c.get(
            f"/v1/sessions/{session_id}/export/{job_id}/download",
            timeout=httpx.Timeout(600.0),
        )
        r.raise_for_status()
        return r.content


# ---------------------------------------------------------------------------
# Job polling
# ---------------------------------------------------------------------------

def get_job(job_id: str) -> dict:
    with _client() as c:
        r = c.get(f"/v1/jobs/{job_id}")
        r.raise_for_status()
        return r.json()


def poll_job(job_id: str, label: str = "Processing...") -> dict:
    """Poll until the job is done.

    Shows a st.progress bar that is removed on completion.
    Returns the job result dict on success, or raises RuntimeError on failure.
    """
    progress_bar = st.progress(0.0, text=label)
    while True:
        data = get_job(job_id)
        record = JobRecord.model_validate(data)
        progress_bar.progress(record.progress, text=record.message or label)
        if record.status == JobStatus.done:
            progress_bar.empty()
            return record.result or {}
        if record.status == JobStatus.error:
            progress_bar.empty()
            raise RuntimeError(record.error or "Job failed")
        time.sleep(0.8)