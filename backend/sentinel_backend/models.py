"""
Shared pydantic contracts — the locked API boundary between backend and frontend.
All request/response shapes are defined here. Import from this module only;
never pass raw rasterio/affine/numpy objects across the HTTP boundary.
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Core geo contract
# ---------------------------------------------------------------------------

class GeoMeta(BaseModel):
    """JSON-safe representation of a rasterio metadata dict."""
    width: int
    height: int
    count: int
    dtype: str
    crs_wkt: str | None = None
    epsg: int | None = None
    transform: list[float]          # [a, b, c, d, e, f] Affine coefficients
    pixel_size: float               # abs(transform.a) in CRS units
    is_geographic: bool

    @classmethod
    def from_rasterio_meta(cls, meta: dict) -> "GeoMeta":
        t = meta["transform"]
        crs = meta.get("crs")
        epsg = None
        if crs:
            try:
                epsg = crs.to_epsg()
            except Exception:
                pass
        return cls(
            width=meta["width"],
            height=meta["height"],
            count=meta["count"],
            dtype=str(meta.get("dtype", "uint8")),
            crs_wkt=crs.to_wkt() if crs else None,
            epsg=epsg,
            transform=[t.a, t.b, t.c, t.d, t.e, t.f],
            pixel_size=abs(t.a),
            is_geographic=crs.is_geographic if crs else False,
        )

    def to_rasterio_meta(self) -> dict:
        from affine import Affine
        meta: dict = {
            "width": self.width,
            "height": self.height,
            "count": self.count,
            "dtype": self.dtype,
            "transform": Affine(*self.transform),
            "driver": "GTiff",
        }
        if self.crs_wkt:
            import rasterio.crs
            meta["crs"] = rasterio.crs.CRS.from_wkt(self.crs_wkt)
        else:
            meta["crs"] = None
        return meta


# ---------------------------------------------------------------------------
# Chip grid contract
# ---------------------------------------------------------------------------

class ChipGridSpec(BaseModel):
    chip_w: int
    chip_h: int
    overlap: float
    edge_mode: str
    naming: str
    n_rows: int
    n_cols: int
    total: int
    windows: list[list[int]]        # [[col_off, row_off, w, h], ...]


# ---------------------------------------------------------------------------
# Job contracts
# ---------------------------------------------------------------------------

class JobStatus(str, Enum):
    queued  = "queued"
    running = "running"
    done    = "done"
    error   = "error"


class JobRecord(BaseModel):
    job_id: str
    status: JobStatus
    progress: float = 0.0           # 0.0 – 1.0
    message: str = ""
    result: dict | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

class SessionInfo(BaseModel):
    session_id: str
    created_at: str
    expires_at: str
    stages_ready: list[str]         # which artifacts exist: stretched/dehazed/enhanced


# ---------------------------------------------------------------------------
# Stage request / result shapes
# ---------------------------------------------------------------------------

class StretchRequest(BaseModel):
    band_indices: list[int] = [1, 2, 3]
    p_low:  float = Field(2.0,  ge=0.0,  le=10.0)
    p_high: float = Field(98.0, ge=90.0, le=100.0)
    per_band: bool = True


class StretchResult(BaseModel):
    artifact: str = "stretched"
    meta: GeoMeta
    stats: dict                     # {mean: [r,g,b], std: [r,g,b]}
    reference_stats_summary: dict | None = None  # {n, mean:[3], std:[3]}


class CloudMaskRequest(BaseModel):
    brightness_thresh: float = Field(0.75, ge=0.5,  le=0.95)
    saturation_thresh: float = Field(0.08, ge=0.01, le=0.20)
    source_stage: str = "stretched"


class CloudMaskResult(BaseModel):
    cloud_pct: float
    cloud_px: int
    total_px: int


class DehazeRequest(BaseModel):
    patch_size:        int   = Field(15,   ge=5,    le=31)
    omega:             float = Field(0.95, ge=0.5,  le=1.0)
    t0:                float = Field(0.1,  ge=0.05, le=0.5)
    use_guided_filter: bool  = True
    mask_clouds:       bool  = True
    brightness_thresh: float = Field(0.75, ge=0.5,  le=0.95)
    saturation_thresh: float = Field(0.08, ge=0.01, le=0.20)


class DehazeResult(BaseModel):
    artifact: str = "dehazed"
    stats: dict                     # {mean: float, std: float}


class EnhanceRequest(BaseModel):
    gray_world:         bool  = False
    ref_norm_enabled:   bool  = False
    ref_norm_method:    str   = "histogram"
    clahe_enabled:      bool  = True
    clahe_clip_limit:   float = Field(2.0, ge=1.0, le=10.0)
    clahe_tile_grid:    list[int] = [8, 8]


class EnhanceResult(BaseModel):
    artifact: str = "enhanced"
    global_stats: dict              # {mean: [r,g,b], std: [r,g,b]}


# ---------------------------------------------------------------------------
# Chip endpoints
# ---------------------------------------------------------------------------

class ChipGridRequest(BaseModel):
    chip_w:    int   = Field(256, ge=64,  le=2048)
    chip_h:    int   = Field(256, ge=64,  le=2048)
    overlap:   float = Field(0.0, ge=0.0, le=0.99)
    edge_mode: str   = "pad"
    naming:    str   = "coords"


class ChipFiltersRequest(BaseModel):
    cloud_enabled:    bool  = True
    cloud_thresh:     float = Field(0.3,   ge=0.0, le=1.0)
    variance_enabled: bool  = True
    variance_thresh:  float = Field(100.0, ge=0.0, le=2000.0)


class ChipFiltersResult(BaseModel):
    accepted: list[int]
    rejected: list[int]
    stats:    list[dict]


class ChipItem(BaseModel):
    index:         int
    row:           int
    col:           int
    thumbnail_url: str
    rejected:      bool


class ChipListResult(BaseModel):
    total:     int
    page:      int
    page_size: int
    items:     list[ChipItem]


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

class ExportRequest(BaseModel):
    include_params:         bool = True
    image_include:          bool = True
    image_format:           str  = "png"
    chips_include:          bool = True
    chips_format:           str  = "png"
    chips_naming:           str  = "coords"
    chips_zscore_normalise: bool = False
    chips_ref_norm_enabled: bool = False
    chips_ref_norm_method:  str  = "histogram"
    chips_include_rejected: bool = False
    annotations_coco:       bool = False
    annotations_yolo:       bool = False


class ExportResult(BaseModel):
    download_url: str
    size_bytes:   int
    n_files:      int


# ---------------------------------------------------------------------------
# Histogram (for cloud overlay chart)
# ---------------------------------------------------------------------------

class HistogramResult(BaseModel):
    bins:       list[float]         # 80 bin edges
    cloud_counts:  list[int]
    land_counts:   list[int]
