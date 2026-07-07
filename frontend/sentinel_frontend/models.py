"""
models.py — Client-side pydantic models for the frontend container.

These are JSON-safe mirrors of the backend models. No rasterio, affine,
or any other heavy geospatial imports here.
"""
from enum import Enum
from typing import Optional
from pydantic import BaseModel


class GeoMeta(BaseModel):
    width: int
    height: int
    count: int
    dtype: str
    crs_wkt: Optional[str] = None
    epsg: Optional[int] = None
    transform: list[float]
    pixel_size: float
    is_geographic: bool


class ChipGridSpec(BaseModel):
    chip_w: int
    chip_h: int
    overlap: float
    edge_mode: str
    naming: str
    n_rows: int
    n_cols: int
    total: int
    windows: list[list[int]]


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    done = "done"
    error = "error"


class JobRecord(BaseModel):
    job_id: str
    status: JobStatus
    progress: float = 0.0
    message: str = ""
    result: Optional[dict] = None
    error: Optional[str] = None


class SessionInfo(BaseModel):
    session_id: str
    created_at: str
    expires_at: str
    stages_ready: list[str]