# Sentinel Imagery Processor — MVP Implementation Plan

## Overview

A Dockerized Streamlit web application that accepts 16-bit Sentinel-2 EO TIFFs,
runs them through a dehazing and enhancement pipeline, chips the result into tiles,
displays those tiles to the user in a paginated grid, and then allows the user to
export them in a chosen format.

---

## Scope for MVP

Included in this build:
- Full preprocessing pipeline (16-bit → uint8)
- Dark Channel Prior dehazing with cloud-adaptive atmospheric light (Improvement #2)
- CLAHE and standardization post-processing
- GDAL-based chipping with parallel export writes (Improvement #5)
- In-app tile viewer: paginated grid display of chips before export
- Export options: PNG, JPEG, GeoTIFF (with georeferencing), NPY
- Streamlit UI with stage-by-stage preview, tile viewer, and format-selectable download
- Docker + Docker Compose packaging

Deferred to future iterations:
- Fast Guided Filter pure-NumPy implementation (#1)
- Post-dehaze white balance / Gray World correction (#3)
- Batch time-series processing mode (#6)
- Interactive chip grid overlay preview (#7)
- Celery + Redis task queue (#8)
- NDVI / band index overlay (#9)
- Cloud Optimized GeoTIFF output (#10)

---

## Directory Structure

```
sentinel-processor/
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
│
├── app/
│   ├── __init__.py
│   ├── main.py                    # Streamlit entry point
│   ├── ui/
│   │   ├── __init__.py
│   │   ├── sidebar.py             # All sidebar parameter widgets
│   │   ├── preview.py             # Stage preview rendering helpers
│   │   └── tile_viewer.py         # Paginated chip grid display component
│   ├── processing/
│   │   ├── __init__.py
│   │   ├── preprocess.py          # 16-bit → uint8 stretch + band selection
│   │   ├── dehazing.py            # DCP + cloud-adaptive atmospheric light
│   │   ├── enhancement.py         # CLAHE + standardization
│   │   └── pipeline.py            # Orchestrator: chains all stages
│   └── chipping/
│       ├── __init__.py
│       ├── gdal_chipper.py        # Chip grid computation + in-memory chip slicing
│       └── tile_exporter.py       # Format-specific export (PNG/JPEG/GeoTIFF/NPY) + zip
│
└── tests/
    ├── test_preprocess.py
    ├── test_dehazing.py
    ├── test_enhancement.py
    └── test_chipping.py
```

---

## Python Dependencies (requirements.txt)

```
# Geospatial (GDAL already present in base Docker image)
rasterio==1.3.10
numpy==1.26.4

# Image processing
opencv-python-headless==4.9.0.80
scikit-image==0.23.2
scipy==1.13.0

# Frontend
streamlit==1.35.0
Pillow==10.3.0

# Utilities
tqdm==4.66.4
```

---

## Docker Setup

### Base image
`osgeo/gdal:ubuntu-small-3.8.4`
Rationale: pre-compiled GDAL with all GeoTIFF/HDF5/JP2 drivers; eliminates version
mismatch issues that plague GDAL pip installs.

### Dockerfile (summary)
- Install python3-pip, libgl1-mesa-glx (OpenCV dep)
- `pip install` requirements.txt excluding GDAL (already in base)
- EXPOSE 8501
- ENTRYPOINT streamlit run app/main.py with maxUploadSize=2048

### docker-compose.yml (summary)
- Single service: sentinel-app
- Port 8501:8501
- Volume mount: ./sample_data:/app/sample_data:ro
- Named volume: chip_output:/tmp/chips

---

## Module Specifications

### processing/preprocess.py

**Purpose**: Read 16-bit Sentinel TIFF and stretch to uint8 RGB.

Key functions:
- `read_sentinel_tiff(path, band_indices) -> (np.ndarray uint16 HWC, meta dict)`
  - Uses rasterio; preserves CRS and transform in meta
  - band_indices are 1-based (rasterio convention)
- `percentile_stretch(arr, p_low=2.0, p_high=98.0, per_band=True) -> np.ndarray uint8`
  - Clips to p_low/p_high percentiles then scales to [0, 255]
  - per_band=True stretches each channel independently (correct for true-color)
  - per_band=False uses global percentiles (preserves relative band ratios)

Design note: Sentinel-2 L2A reflectance values cluster in 0–3000 DN despite 0–65535
range. Naive linear stretch produces near-black output. Percentile clipping is mandatory.

---

### processing/dehazing.py

**Purpose**: Remove atmospheric haze using Dark Channel Prior, with cloud-adaptive
atmospheric light estimation.

Key functions:
- `dark_channel(img_f32, patch_size=15) -> np.ndarray`
  - img_f32: float32 (H,W,3) in [0,1]
  - Min over channels then spatial erosion (minimum filter)
- `detect_clouds_simple(img_uint8, brightness_thresh=0.75, saturation_thresh=0.08) -> bool mask`
  - Brightness: mean of all channels > threshold
  - Saturation: max-min across channels < threshold (clouds are nearly white)
  - Returns (H,W) boolean mask; True = cloud pixel
- `estimate_atmospheric_light(img_f32, dark, top_percent=0.001, cloud_mask=None) -> np.ndarray (3,)`
  - Excludes cloud pixels from candidate set
  - Falls back to full image if >80% of scene is cloud
  - Selects highest-intensity pixel from top 0.1% bright dark-channel candidates
- `transmission_map(img_f32, A, patch_size=15, omega=0.95) -> np.ndarray`
  - t(x) = 1 - omega * dark_channel(img / A)
- `guided_filter_transmission(guide, transmission, radius=60, eps=1e-3) -> np.ndarray`
  - Uses cv2.ximgproc if available; falls back to bilateral filter
- `recover_scene_radiance(img_f32, t, A, t0=0.1) -> np.ndarray`
  - J(x) = (I(x) - A) / max(t(x), t0) + A
- `dehaze(img_uint8, patch_size, omega, t0, use_guided_filter, mask_clouds=True) -> np.ndarray uint8`
  - Full pipeline: detect clouds → dark channel → A → transmission → refine → recover
  - Restores original (pre-dehaze) pixel values at cloud locations in output
  - Cloud pixels are opaque objects DCP cannot fix; leaving them intact avoids artifacts

---

### processing/enhancement.py

**Purpose**: Improve contrast and visual quality of dehazed imagery.

Key functions:
- `apply_clahe(img_uint8, clip_limit=2.0, tile_grid_size=(8,8)) -> np.ndarray uint8`
  - Converts RGB → LAB, applies CLAHE to L channel only, converts back
  - Per-channel RGB CLAHE causes hue shifts; LAB preserves color fidelity
- `apply_standardization(img_uint8, target_mean=127.5, target_std=45.0) -> np.ndarray uint8`
  - Z-score normalize then rescale to target distribution
  - Useful for downstream ML pipeline inputs

---

### processing/pipeline.py

**Purpose**: Orchestrates all processing stages; holds config dataclass.

Key types:
- `PipelineConfig` dataclass: all parameters for every stage
- `PipelineResult` dataclass: final image (uint8 HWC), rasterio meta, stages dict
  (intermediate images keyed by stage name for UI preview)

Key function:
- `run_pipeline(tiff_path, config) -> PipelineResult`
  - Stage 1: read + stretch → stores "preprocessed" in stages
  - Stage 2: dehaze (if enabled) → stores "dehazed" in stages
  - Stage 3: enhance → stores "enhanced" in stages
  - Updates meta: dtype=uint8, compress=lzw, predictor=2

---

### chipping/gdal_chipper.py

**Purpose**: Compute the chip grid and slice the processed image into in-memory chip
arrays. Does NOT write to disk — that is tile_exporter.py's responsibility.

Key types:
- `ChipGrid` dataclass:
  - `windows: List[Tuple[int,int,int,int]]`  — (col_off, row_off, w, h) per chip
  - `source_image: np.ndarray`               — full processed uint8 (H,W,3) image
  - `source_meta: dict`                      — rasterio meta with CRS + transform
  - `chip_w: int`, `chip_h: int`
  - `n_rows: int`, `n_cols: int`
  - `total: int`                             — len(windows)

Key functions:
- `compute_chip_grid(img_w, img_h, chip_w, chip_h, overlap=0.0) -> List[(col, row, w, h)]`
  - overlap is a fraction (0.1 = 10% overlap between adjacent chips)
  - Edge chips are padded to full chip_w × chip_h with zeros
- `build_chip_grid(processed_img, source_meta, chip_w, chip_h, overlap) -> ChipGrid`
  - Constructs a ChipGrid; stores source_image as a reference (not copies)
  - Memory note: does NOT materialise per-chip arrays; all display/export accesses
    slice from source_image on demand to keep memory at O(1 image) not O(N chips)
- `get_chip(grid, index) -> Tuple[np.ndarray, dict]`
  - Returns (chip_array uint8 HWC, chip_meta) for chip at flat index
  - chip_meta includes per-chip affine transform derived from source transform
  - chip_array is padded to full chip_w × chip_h at edges
- `chip_affine(source_transform, col_off, row_off) -> Affine`
  - Computes per-chip geotransform:
    origin_x = transform.c + col_off * transform.a
    origin_y = transform.f + row_off * transform.e

---

### chipping/tile_exporter.py

**Purpose**: Write chips from a ChipGrid to disk in a chosen format using parallel
workers, then zip for download.

Supported formats: "png", "jpeg", "geotiff", "npy"

Key functions:
- `_export_single_chip(args) -> str`
  - Top-level picklable worker function (required for multiprocessing)
  - args: (chip_array, chip_meta, out_path, fmt)
  - PNG/JPEG: PIL Image.save() — no georeferencing embedded
  - GeoTIFF: rasterio.open(...) with full affine + CRS from chip_meta
  - NPY: np.save() — raw array, useful for ML pipeline ingestion
- `export_chips(grid, output_dir, fmt, naming) -> List[str]`
  - Materialises all chip arrays via get_chip(), dispatches via ProcessPoolExecutor
  - Returns list of written paths
  - naming: "rowcol" → chip_r0001_c0002.{ext}; "coords" → chip_X_Y.{ext}
- `zip_export(chip_paths) -> bytes`
  - Returns in-memory ZIP bytes for Streamlit st.download_button

Performance note: ProcessPoolExecutor is safe here because each worker operates on
an independent chip array (passed by value via pickling). Expected 3–6x speedup.

---

### ui/tile_viewer.py

**Purpose**: Streamlit component for displaying the chip grid as a paginated thumbnail
gallery, with per-chip expand-on-click and export controls.

Key functions:
- `render_tile_viewer(grid: ChipGrid, page_size: int = 16)`
  - Displays chips in a CSS-grid-style layout using st.columns
  - Pagination: prev/next buttons stored in st.session_state["tile_page"]
  - page_size: number of chips per page (default 16 = 4×4 grid)
  - Each cell shows a thumbnail (downsampled to 128×128 for display speed)
  - st.expander on each cell shows the full-resolution chip on click
  - Shows chip index, row/col, and pixel coordinates as caption
- `render_export_controls(grid: ChipGrid) -> bytes | None`
  - Format selector: PNG / JPEG / GeoTIFF / NPY
  - Chip naming selector: rowcol / coords
  - "Export All Chips" button — triggers export_chips() + zip_export()
  - Returns zip bytes to pass to st.download_button in main.py
  - Shows a progress spinner during export

Thumbnail generation note: downsampling to 128×128 for display is done with
PIL.Image.LANCZOS in get_chip() output — never stored in session_state to avoid
bloating it with thumbnail arrays.

---

### app/main.py

**Purpose**: Streamlit entry point — file upload, parameter controls, stage previews,
tile viewer, and export.

UI layout (two distinct phases):

**Phase A — Processing** (shown after upload, before pipeline run):
- Sidebar: band selection, preprocessing params, dehazing toggle + params (including
  cloud detection thresholds), enhancement selector + params, chipping params
  (chip size, overlap, naming)
- Main panel: file uploader → "Run Pipeline" button → progress bar →
  3-column stage preview (preprocessed | dehazed | enhanced)

**Phase B — Tile Viewer + Export** (shown after pipeline completes):
- Full-width tile viewer via render_tile_viewer(grid)
- Below viewer: render_export_controls(grid) → format selector → download button
- "Re-run Pipeline" button returns to Phase A with params intact

Key behaviors:
- st.session_state["chip_grid"] holds the ChipGrid after pipeline runs (no re-run on
  widget interaction)
- st.session_state["pipeline_result"] holds PipelineResult for stage preview
- Progress bar: read (20%) → dehaze (50%) → enhance (70%) → chip grid built (85%) →
  viewer ready (100%)
- Temp TIFF file cleaned up immediately after pipeline completes (before tile viewer)
- Export zip written to BytesIO (no disk write in container)

---

## Data Flow

```
TIFF (16-bit, 3-band)
       │
       ▼
preprocess.py: percentile_stretch() ──────────────────► uint8 (H,W,3)
       │                                                  [stage: "preprocessed"]
       ▼
dehazing.py:  detect_clouds() → dark_channel() → A
              → transmission() → guided_filter()
              → recover_radiance() → restore_clouds() ──► uint8 (H,W,3)
       │                                                  [stage: "dehazed"]
       ▼
enhancement.py: apply_clahe() or apply_standardization() ► uint8 (H,W,3)
       │                                                  [stage: "enhanced"]
       ▼
gdal_chipper.py: build_chip_grid()
                 → ChipGrid (windows + source_image ref)  ◄─ no disk writes yet
       │
       ▼
ui/tile_viewer.py: render_tile_viewer(grid)
                 → paginated thumbnail gallery
                 → expand individual chip on click
       │
       ▼  [user selects format + clicks Export]
tile_exporter.py: export_chips()
                 → ProcessPoolExecutor(_export_single_chip)
                 → PNG / JPEG / GeoTIFF / NPY files
                 → zip_export() ──────────────────────► bytes
       │
       ▼
st.download_button()
```

The rasterio `meta` dict (carrying `crs` and `transform`) is passed through every
stage unchanged. Only `dtype`, `count`, and compression fields are updated at the end.
ChipGrid carries source_meta so each chip's affine geotransform can be computed on
demand at export time, preserving full georeferencing for GeoTIFF export.

---

## Implementation Order and Agent Assignment

| Phase | Agent | Deliverables | Can start after |
|-------|-------|--------------|-----------------|
| 1 | Infrastructure | Dockerfile, docker-compose.yml, requirements.txt, all __init__.py + directory scaffold | Nothing — runs first |
| 2 | Preprocessing | preprocess.py, test_preprocess.py | Phase 1 |
| 3 | Dehazing + Enhancement | dehazing.py, enhancement.py, pipeline.py, test_dehazing.py, test_enhancement.py | Phase 2 (needs preprocess API) |
| 4 | Tile Viewer + Export | gdal_chipper.py, tile_exporter.py, ui/tile_viewer.py, test_chipping.py | Phase 3 (needs PipelineResult shape) |
| 5 | UI | main.py, ui/sidebar.py, ui/preview.py | Phases 3 + 4 (needs all APIs finalised) |
| 6 | Review | Cross-module correctness, API contract verification, test coverage audit | Phase 5 |

---

## Critical Correctness Constraints

1. Band indices are 1-based in rasterio; never pass 0-indexed values to `src.read()`
2. The `meta["transform"]` affine matrix must be carried through ALL stages unmodified
3. Per-chip geotransform = source transform with origin shifted by (col_off, row_off) in
   pixel units: `origin_x = transform.c + col_off * transform.a`
4. CLAHE must operate on L channel in LAB space, not per-channel RGB
5. Cloud pixel restoration in dehaze() must happen AFTER guided filter refinement,
   not before (the filter may smear cloud edges into land pixels)
6. `_export_single_chip` must be a top-level module function (not a lambda or nested
   function) to be picklable by multiprocessing
7. Chip arrays must NOT be stored in st.session_state — only the ChipGrid (which holds
   a reference to source_image). Storing per-chip arrays in session_state would multiply
   memory usage by the chip count.
8. Thumbnail downsampling for display must happen at render time in tile_viewer.py,
   never persisted to session_state

---

## Future Iteration Backlog

1. Fast Guided Filter in pure NumPy (eliminate ximgproc dep)
2. Post-dehaze Gray World white balance correction
3. Batch time-series processing mode
4. Interactive chip grid overlay preview in UI
5. Celery + Redis task queue for multi-user production deployment
6. NDVI / NDWI band index overlay and chip metadata
7. Cloud Optimized GeoTIFF (COG) output format
