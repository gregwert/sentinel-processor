# Sentinel Processor — Iteration 2 Plan

## What changed from Iteration 1

The app evolves from a single-form batch processor into a step-by-step data creation
editor. Users move through a linear workflow with side-by-side comparisons and live
feedback at each stage rather than configuring everything upfront and running blind.

---

## Scope

### In this iteration
- Step-based wizard UI (replaces sidebar form + single Run button)
- Cloud overlay visualisation with live-adjustable thresholds
- DCP dehazing output shown as side-by-side comparison
- CLAHE applied to full image as before; standardisation moved to export time
- Chip grid overlay on processed image before running chipping
- Chip size in pixels OR metres (converted via rasterio geotransform)
- Export-time normalisation: standardisation using full-image global stats

### Deferred to future iterations
- Histogram popup comparing cloud vs non-cloud pixel distributions
- COCO / YOLO annotation export
- Session management (save / reload configs and in-progress work)
- Per-chip quality filtering (cloud %, variance threshold)
- Train / val / test split assignment

---

## New directory structure

Changes from Iteration 1 are marked with (+new) or (~modified):

```
app/
├── main.py                          (~modified — step orchestrator)
├── ui/
│   ├── __init__.py
│   ├── preview.py                   (unchanged)
│   ├── tile_viewer.py               (~modified — add global-stats export option)
│   ├── cloud_overlay.py             (+new — cloud mask composite rendering)
│   ├── grid_overlay.py              (+new — chip grid composite rendering)
│   └── steps/
│       ├── __init__.py              (+new)
│       ├── step_upload.py           (+new — upload + immediate stretch preview)
│       ├── step_dehaze.py           (+new — cloud overlay, threshold sliders, DCP run)
│       ├── step_enhance.py          (+new — CLAHE controls, before/after, run)
│       └── step_chip.py             (+new — grid overlay, chip size px/m, run)
├── processing/
│   ├── preprocess.py                (unchanged)
│   ├── dehazing.py                  (unchanged)
│   ├── enhancement.py               (unchanged)
│   └── pipeline.py                  (~modified — remove standardisation from main
│                                     pipeline; it moves to export)
└── chipping/
    ├── gdal_chipper.py              (~modified — add metres→pixels conversion)
    └── tile_exporter.py             (~modified — add global-stats standardisation
                                      at export time)
```

---

## Workflow and step definitions

The UI is driven by `st.session_state["step"]` which is one of:
`"upload"` → `"dehaze"` → `"enhance"` → `"chip"` → `"review"`

Each step renders its own layout. A breadcrumb/progress bar at the top shows the user
where they are. Completed steps show a summary line (e.g. "Dehaze: omega=0.95,
clouds masked"). Each step has a "← Back" button that returns to the previous step
without losing work (all computed results are held in session_state).

### session_state keys

| Key | Type | Set at |
|-----|------|--------|
| `step` | str | throughout |
| `stretched_image` | np.ndarray uint8 (H,W,3) | step_upload |
| `source_meta` | dict | step_upload |
| `cloud_mask` | np.ndarray bool (H,W) | step_dehaze (live) |
| `dehazed_image` | np.ndarray uint8 (H,W,3) | step_dehaze (after run) |
| `enhanced_image` | np.ndarray uint8 (H,W,3) | step_enhance (after run) |
| `global_stats` | dict {mean, std per channel} | step_enhance (after run) |
| `chip_grid` | ChipGrid | step_chip (after run) |
| `upload_params` | dict | step_upload |
| `dehaze_params` | dict | step_dehaze |
| `enhance_params` | dict | step_enhance |
| `chip_params` | dict | step_chip |

---

## Module specifications

---

### app/ui/cloud_overlay.py  (+new)

**Purpose**: Render a semi-transparent cloud mask composite over the stretched image
for real-time threshold feedback.

```python
def render_cloud_composite(
    img_uint8: np.ndarray,         # (H,W,3) uint8 — stretched image
    cloud_mask: np.ndarray,        # (H,W) bool
    overlay_colour: tuple = (255, 0, 0),   # RGB — default red
    alpha: float = 0.45,
    max_display_px: int = 800,     # downsample long edge to this for display speed
) -> PIL.Image:
```

Implementation:
- Downsample img_uint8 to max_display_px on the long edge (PIL LANCZOS) for display
- Downsample cloud_mask by same factor (nearest-neighbour via PIL NEAREST)
- Create RGBA composite: base image as RGB layer, overlay_colour at alpha where
  cloud_mask is True, transparent elsewhere
- Return PIL Image (RGBA)

```python
def compute_cloud_stats(cloud_mask: np.ndarray) -> dict:
    """Return {cloud_pct, cloud_px, total_px} for display as metrics."""
```

---

### app/ui/grid_overlay.py  (+new)

**Purpose**: Render the chip grid as line overlays on the processed image.

```python
def render_grid_composite(
    img_uint8: np.ndarray,         # (H,W,3) uint8 — enhanced image
    grid: ChipGrid,
    line_colour: tuple = (255, 255, 0),    # RGB yellow
    line_width: int = 1,
    max_display_px: int = 900,
) -> PIL.Image:
```

Implementation:
- Downsample image to max_display_px on long edge
- Compute display scale factor: scale = display_w / original_w
- Use PIL.ImageDraw to draw grid lines at scaled positions:
  - Vertical lines at each unique col_off * scale
  - Horizontal lines at each unique row_off * scale
  - Also draw the right/bottom edges
- Return PIL Image (RGB)

```python
def chip_size_metres_to_pixels(
    metres: float,
    source_meta: dict,
) -> int:
    """
    Convert a chip dimension from metres to pixels using the rasterio geotransform.
    For projected CRS (UTM etc): pixel_size = abs(transform.a) in metres — exact.
    For geographic CRS (degrees): approximate using 111320 m/degree at equator,
    adjusted by cos(mean_latitude) for longitude. Warns in return value if geographic.
    Returns (pixels: int, is_approximate: bool).
    """
```

---

### app/ui/steps/step_upload.py  (+new)

**Purpose**: Step 1 — file upload, band selection, percentile params, immediate
stretch preview.

```python
def render(state: dict) -> bool:
    """Render upload step. Returns True when user advances to next step."""
```

Layout:
- File uploader (tif/tiff, 2 GB limit)
- When file uploaded: immediately run `read_sentinel_tiff` + `percentile_stretch`
  and store result in session_state — no button press needed
- Show stretched image preview (full width, downsampled)
- Expander "Band & stretch settings":
  - Band R/G/B indices (default 1,2,3)
  - Percentile low/high sliders
  - Per-band stretch checkbox (default False)
  - Any change re-runs stretch immediately (reactive)
- Show image stats: dimensions, band count, CRS, pixel size
- "Next: Dehazing →" button advances step

---

### app/ui/steps/step_dehaze.py  (+new)

**Purpose**: Step 2 — cloud overlay with live threshold feedback, DCP parameters,
run dehazing, side-by-side before/after.

```python
def render(state: dict) -> bool:
    """Render dehaze step. Returns True when user advances."""
```

Layout (two phases within this step):

**Phase A — Cloud configuration** (before "Run Dehazing" is clicked):
- Toggle "Enable cloud masking" (default True)
- If enabled: two-column layout
  - Left: stretched image (static, full step width ÷ 2)
  - Right: cloud composite from `render_cloud_composite()`, updates live on slider change
- Below image pair: cloud threshold sliders
  - Brightness threshold (0.5–0.95, default 0.75)
  - Saturation threshold (0.01–0.20, default 0.08)
- Cloud stats line: "X% of scene detected as cloud (Y,000 px)"
- Expander "DCP parameters":
  - Omega slider (0.5–1.0, default 0.95)
  - t₀ slider (0.05–0.5, default 0.10)
  - Patch size slider (5–31, default 15)
  - Guided filter checkbox (default True)
- Toggle "Enable dehazing" — if off, skip directly with stretched_image as output
- "Run Dehazing" button

**Phase B — After "Run Dehazing"** (dehazed_image in session_state):
- Two-column side-by-side:
  - Left: stretched image labelled "Before"
  - Right: dehazed image labelled "After DCP"
- Metric row: before/after mean and std per image
- "Re-run with different settings" button → clears dehazed_image, returns to Phase A
- "Next: Enhancement →" button

Key behaviour: threshold sliders recompute `detect_clouds_simple` and
`render_cloud_composite` reactively on each Streamlit rerun. This is fast because
`detect_clouds_simple` is O(H×W) and the composite renders a downsampled image.
The full DCP dehazing only runs when "Run Dehazing" is clicked.

---

### app/ui/steps/step_enhance.py  (+new)

**Purpose**: Step 3 — CLAHE on full image, before/after comparison.

```python
def render(state: dict) -> bool:
    """Render enhance step. Returns True when user advances."""
```

Layout:
- Enhancement method selector: "CLAHE" | "None"
  - Standardisation is removed from this step (moved to export)
- If CLAHE:
  - Clip limit slider (0.5–10.0, default 2.0)
  - Tile grid size selector (4/8/16/32, default 8)
- "Apply Enhancement" button
- After run: side-by-side before (dehazed) / after (enhanced)
- Global stats computed here and stored: mean and std per channel of enhanced_image
  (used later for export-time standardisation)
- "Re-run" button
- "Next: Chipping →" button

Note: if enhancement = "None", enhanced_image = dehazed_image (no processing),
global_stats still computed.

---

### app/ui/steps/step_chip.py  (+new)

**Purpose**: Step 4 — chip size configuration with grid overlay, run chipping.

```python
def render(state: dict) -> bool:
    """Render chip step. Returns True when user advances to review."""
```

Layout (two phases):

**Phase A — Grid configuration**:
- Chip size unit toggle: "Pixels" | "Metres"
- If Pixels:
  - Width input (64–2048, default 256, step 64)
  - Height input (64–2048, default 256, step 64)
- If Metres:
  - Width metres input (float, default 1000.0)
  - Height metres input (float, default 1000.0)
  - Converts to pixels via `chip_size_metres_to_pixels()`
  - Shows computed pixel size below: "≈ 256 × 256 px"
  - If geographic CRS: show warning "Approximate — scene uses geographic CRS"
- Overlap fraction slider (0.0–0.49, default 0.0)
- Chip naming selector: rowcol | coords
- Grid overlay updates reactively as parameters change (recomputes chip grid
  geometry and rerenders overlay — no full pipeline re-run needed)
- Shows: "Grid: N rows × M cols = K chips"
- "Run Chipping" button

**Phase B — After chipping**:
- Grid overlay image shown (confirmation)
- Chip count summary
- "Proceed to Review →" button advances to tile viewer / export

---

### app/chipping/gdal_chipper.py  (~modified)

Add `chip_size_metres_to_pixels` — but this lives in `grid_overlay.py` (UI layer).
No changes needed to gdal_chipper.py itself; it already accepts pixel dimensions.

---

### app/chipping/tile_exporter.py  (~modified)

Add export-time standardisation using global image stats:

```python
def export_chips(
    grid,
    output_dir: str,
    fmt: str = "png",
    naming: str = "rowcol",
    normalise: bool = False,
    global_stats: dict = None,   # {mean: (3,), std: (3,)} from full enhanced image
) -> List[str]:
```

In `_export_single_chip`, if `normalise=True` and `global_stats` provided:
- Apply z-score: `(chip.astype(float32) - mean) / (std + 1e-6)`
- Rescale to [0, 255] uint8
- This is applied after format-specific conversion, only for PNG/JPEG/GeoTIFF
- NPY export gets the raw normalised float32 array (not clipped to uint8) since
  ML pipelines consume float arrays directly

---

### app/ui/tile_viewer.py  (~modified)

Add normalisation option to `render_export_controls`:
```python
def render_export_controls(grid, global_stats=None) -> bytes | None:
```
- If `global_stats` is not None: show "Normalise chips (z-score)" checkbox
- Pass normalise flag and global_stats through to export_chips

---

### app/main.py  (~modified)

Replace the Phase A / Phase B session_state pattern with a step router:

```python
STEPS = ["upload", "dehaze", "enhance", "chip", "review"]

def render_breadcrumb(current_step: str): ...

step = st.session_state.get("step", "upload")
render_breadcrumb(step)

if step == "upload":
    from ui.steps.step_upload import render
    if render(st.session_state): st.session_state["step"] = "dehaze"; st.rerun()

elif step == "dehaze":
    from ui.steps.step_dehaze import render
    if render(st.session_state): st.session_state["step"] = "enhance"; st.rerun()

elif step == "enhance":
    from ui.steps.step_enhance import render
    if render(st.session_state): st.session_state["step"] = "chip"; st.rerun()

elif step == "chip":
    from ui.steps.step_chip import render
    if render(st.session_state): st.session_state["step"] = "review"; st.rerun()

elif step == "review":
    render_tile_viewer(st.session_state["chip_grid"])
    st.divider()
    zip_bytes = render_export_controls(
        st.session_state["chip_grid"],
        global_stats=st.session_state.get("global_stats")
    )
    if zip_bytes:
        st.download_button(...)
```

sidebar.py is no longer used — delete it (or keep as dead code with a deprecation
comment if you want to preserve git history cleanly).

---

### pipeline.py  (~modified)

Remove `std_target_mean`, `std_target_std` from PipelineConfig (standardisation
no longer runs as a pipeline stage). Remove `"standardization"` branch from
`run_pipeline`. Enhancement is now CLAHE or None only.

---

## Data flow (updated)

```
Upload TIFF
    │
    ▼ (immediate, reactive)
preprocess.py: percentile_stretch() ──────────────► stretched_image (H,W,3) uint8
    │                                                saved to session_state
    ▼ (on "Run Dehazing" click)
cloud_overlay.py: detect_clouds_simple() ──────────► cloud_mask (live preview only)
dehazing.py: dehaze() ─────────────────────────────► dehazed_image (H,W,3) uint8
    │
    ▼ (on "Apply Enhancement" click)
enhancement.py: apply_clahe() ─────────────────────► enhanced_image (H,W,3) uint8
                                                      + global_stats {mean, std}
    │
    ▼ (reactive on param change — no processing)
grid_overlay.py: render_grid_composite() ──────────► overlay preview (display only)

    │ (on "Run Chipping" click)
    ▼
gdal_chipper.py: build_chip_grid() ────────────────► ChipGrid (in-memory)
    │
    ▼ (on "Export" click)
tile_exporter.py: export_chips(normalise=...) ─────► files → zip → download
```

---

## Implementation order and agent assignment

| Phase | Agent | Owns | Depends on |
|-------|-------|------|-----------|
| 1 | Infrastructure | Create `app/ui/steps/` dir + stubs for all new files | Nothing |
| 2 | Overlay components | `cloud_overlay.py`, `grid_overlay.py` (incl. metres conversion) | Phase 1 |
| 3 | Step UI | `step_upload.py`, `step_dehaze.py`, `step_enhance.py`, `step_chip.py` | Phase 2 |
| 4 | Main + pipeline cleanup | `main.py` refactor, `pipeline.py` simplification, `sidebar.py` removal | Phase 3 |
| 5 | Export normalisation | `tile_exporter.py` + `tile_viewer.py` changes | Phase 1 |
| 6 | Reviewer | Full cross-module audit | Phase 4 + 5 |

Phases 4 and 5 can run in parallel once Phase 3 is done.

---

## Critical correctness constraints

1. Cloud mask computation in step_dehaze MUST use the stored `stretched_image` from
   session_state, not re-read the TIFF (which has been deleted after upload)
2. Metres→pixels conversion must flag approximate results when CRS is geographic
3. Grid overlay rerender must NOT re-run dehaze or CLAHE — it only recomputes the
   chip window geometry and redraws lines on the stored enhanced_image
4. global_stats must be computed from enhanced_image BEFORE it is displayed in the
   tile viewer — this is the reference for export normalisation
5. "Re-run" buttons in each step must clear only that step's session_state key
   (not downstream keys) so the user can go back without losing later work

---

## Future iteration backlog

- Histogram popup: cloud vs non-cloud pixel distribution comparison with
  transmission map overlay
- COCO / YOLO annotation export with chip manifest CSV
- Session management: save/load processing configs as named presets; resume
  in-progress work
- Per-chip quality filtering: auto-reject chips below variance threshold or
  above cloud coverage threshold
- Train / val / test split assignment in tile viewer
