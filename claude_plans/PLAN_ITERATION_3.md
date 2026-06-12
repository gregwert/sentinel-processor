# Sentinel Processor — Iteration 3 Plan

**Prepared:** 2026-06-11
**Scope:** 7 features across Groups A (quick wins), B (medium scope), C (large)

---

## Architecture Overview (Current State)

The app is a five-step Streamlit wizard (`upload → dehaze → enhance → chip → review`) routed through `app/main.py`. All step modules follow a uniform `render(state: dict) -> bool` convention. Processing logic lives in `app/processing/`, geometry in `app/chipping/`, and display helpers in `app/ui/`.

---

## Group A — Quick Wins

---

### Feature 1: Breadcrumb / Progress Indicator (Refactor)

A `render_breadcrumb(current: str)` already exists in `app/main.py`. The task is to move it into the shared step helpers module.

#### Files Changed

**`app/ui/steps/__init__.py`** — Add `render_breadcrumb`, `STEPS`, `STEP_LABELS`.

```python
STEPS = ["upload", "dehaze", "enhance", "chip", "review"]
STEP_LABELS = {
    "upload":  "1. Upload",
    "dehaze":  "2. Dehaze",
    "enhance": "3. Enhance",
    "chip":    "4. Chip",
    "review":  "5. Review",
}

def render_breadcrumb(current: str) -> None:
    """Render the step progress bar at the top of every wizard page."""
```

**`app/main.py`** — Remove local `render_breadcrumb`, `STEPS`, `STEP_LABELS`. Import from `ui.steps`. Call site `render_breadcrumb(step)` stays identical.

#### New Session State Keys
None.

#### Edge Cases
- `main.py` uses `STEPS.index(current)` for routing — after the move it must import `STEPS` from `ui.steps`.
- No circular import risk: `__init__.py` does not import from individual step modules.

---

### Feature 2: Per-Chip Quality Filtering

#### New File: `app/chipping/chip_filter.py`

```python
def compute_chip_stats(
    chip_array: np.ndarray,
    cloud_mask: np.ndarray | None,
    col_off: int,
    row_off: int,
    chip_w: int,
    chip_h: int,
) -> dict:
    """Compute cloud_pct and variance for one chip window.

    Returns dict with keys: cloud_pct (float 0–1), variance (float).
    """
```

```python
def apply_chip_filters(
    grid: "ChipGrid",
    cloud_mask: np.ndarray | None,
    cloud_thresh: float = 0.3,
    variance_thresh: float = 100.0,
    enable_cloud_filter: bool = True,
    enable_variance_filter: bool = True,
) -> tuple[list[int], list[int], list[dict]]:
    """Evaluate quality filters across all chips.

    Returns (accepted_indices, rejected_indices, chip_stats).
    chip_stats is a list[dict] with one entry per chip: chip_index,
    cloud_pct, variance, rejected.
    """
```

**Key notes:**
- Slice `cloud_mask[row_off:row_off+chip_h, col_off:col_off+chip_w]`; use actual slice size as denominator (handles edge chips).
- Variance: `chip_array.astype(np.float32).var()` — catches all-zero padding and featureless chips.
- Both functions must be importable without Streamlit.

#### Files Changed

**`app/ui/steps/step_chip.py`** — Add quality filter expander in Phase B (after `st.success`, before grid overlay):

```
with st.expander("Quality Filters", expanded=False):
    enable_cloud  = st.toggle("Cloud coverage filter", ...)
    cloud_thresh  = st.slider("Max cloud fraction", 0.0, 1.0, 0.30, 0.01, ...)
    enable_var    = st.toggle("Variance filter", ...)
    var_thresh    = st.slider("Min variance", 0.0, 2000.0, 100.0, 10.0, ...)
    show_rejected = st.checkbox("Show rejected chips in viewer", ...)
```

When any filter is on, call `apply_chip_filters(...)` and write to state:

```python
state["chip_quality"] = {
    "accepted": accepted_indices,
    "rejected": rejected_indices,
    "stats": chip_stats,
    "cloud_thresh": cloud_thresh,
    "variance_thresh": var_thresh,
    "enable_cloud_filter": enable_cloud,
    "enable_variance_filter": enable_var,
    "show_rejected": show_rejected,
}
```

Pass into tile viewer: `render_tile_viewer(grid, quality=state.get("chip_quality"))`.

Also clear `chip_quality` when Re-chip button is clicked.

**`app/ui/tile_viewer.py`** — Update signature:

```python
def render_tile_viewer(grid, page_size=DEFAULT_PAGE_SIZE, quality: dict | None = None) -> None:
```

- When `show_rejected` is False: skip rejected chips when building page.
- When `show_rejected` is True: render them with a red tint overlay.
- Add header caption showing accepted / rejected counts when filters are active.

```python
def _tint_chip(chip_arr: np.ndarray, colour=(200, 0, 0), alpha=0.35) -> np.ndarray:
    """Blend a solid colour over a chip array to visually mark it as rejected."""
```

**`app/chipping/tile_exporter.py`** — Add parameters to `export_chips`:

```python
def export_chips(
    grid, output_dir, fmt="png", naming="rowcol",
    normalise=False, global_stats=None, config=None,
    rejected_indices: list[int] | None = None,
    include_rejected: bool = False,
) -> list[str]:
```

**`app/ui/steps/step_review.py`** — Extract `rejected_indices` from `state.get("chip_quality", {})`. Add checkbox "Include rejected chips in export" (visible only when rejected chips exist). Pass through to exporter.

#### New Session State Keys
- `chip_quality : dict` — keys: `accepted`, `rejected`, `stats`, `cloud_thresh`, `variance_thresh`, `enable_cloud_filter`, `enable_variance_filter`, `show_rejected`.

#### Edge Cases
- When `cloud_mask` absent (dehaze skipped): `cloud_pct = 0.0`, cloud filter should show a warning or be disabled.
- When neither filter enabled: write `chip_quality` with all chips accepted, empty rejected.
- Clear `chip_quality` in the Re-chip button handler alongside `chip_grid`.
- Edge chip cloud mask slice: use `slice.size` not nominal `chip_w * chip_h` as denominator.

---

### Feature 3: Chip Manifest CSV

#### New File: `app/chipping/manifest.py`

```python
def chip_lat_lon_bounds(
    transform: "affine.Affine",
    crs: "rasterio.crs.CRS | None",
    col_off: int, row_off: int,
    chip_w: int, chip_h: int,
) -> tuple[float, float, float, float]:
    """Return (lon_min, lat_min, lon_max, lat_max) for a chip window.

    Geographic CRS: read directly from affine values.
    Projected CRS: reproject corners via pyproj to EPSG:4326.
    CRS None: return (None, None, None, None).
    """
```

```python
def build_manifest(
    grid: "ChipGrid",
    chip_stats: list[dict] | None = None,
    naming: str = "rowcol",
    fmt_ext: str = ".png",
) -> list[dict]:
    """Build per-chip manifest row dicts.

    Columns: chip_index, row, col, pixel_x_min, pixel_y_min, pixel_x_max,
    pixel_y_max, lon_min, lat_min, lon_max, lat_max, cloud_pct, variance,
    filename, rejected.

    When chip_stats is None: cloud_pct=0.0, variance=0.0, rejected=False.
    """
```

```python
def write_manifest_csv(rows: list[dict]) -> bytes:
    """Serialise manifest rows to UTF-8 CSV bytes (no filesystem write)."""
```

#### Files Changed

**`app/ui/steps/step_review.py`** — In `_build_export_zip`, after chip files are written:

```python
from chipping.manifest import build_manifest, write_manifest_csv

rows = build_manifest(state["chip_grid"], chip_stats, chip_naming, fmt_ext)
if not include_rejected:
    rejected_set = set(state.get("chip_quality", {}).get("rejected", []))
    rows = [r for r in rows if r["chip_index"] not in rejected_set or not r["rejected"]]
zf.writestr("chips/manifest.csv", write_manifest_csv(rows))
added = True
```

#### Dependencies
- Feature 2 first (provides `chip_stats`); `build_manifest` accepts `chip_stats=None` for graceful fallback.
- `filename` column must use the same naming logic as `tile_exporter.py` — import `_coords_filename` from there.
- `pyproj` is a transitive dependency of rasterio; already present.

#### Edge Cases
- When chipping was skipped: no manifest written.
- Rejected chips still appear in manifest with `rejected=true`; export flag controls whether chip files are included.
- CRS None: set lon/lat fields to empty string.

---

## Group B — Medium Scope

---

### Feature 4: Histogram View in Dehaze Step

#### Files Changed

**`app/ui/steps/step_dehaze.py`** — Add expander inside `if mask_clouds:` block, after the cloud stats caption:

```python
if cloud_mask.any() and (~cloud_mask).any():
    with st.expander("Pixel brightness histogram"):
        _render_brightness_histogram(state["stretched_image"], cloud_mask)
else:
    st.caption("Histogram unavailable — all pixels in one class.")
```

#### New Function in `app/ui/steps/step_dehaze.py`

```python
def _render_brightness_histogram(img_uint8: np.ndarray, cloud_mask: np.ndarray) -> None:
    """Render overlapping Altair histograms: cloud pixels (red) vs land (blue).

    Subsamples to 200 000 pixels for performance on large images.
    """
    import altair as alt
    import pandas as pd

    brightness = img_uint8.mean(axis=2).flatten()
    label = np.where(cloud_mask.flatten(), "Cloud", "Land")
    n = len(brightness)
    if n > 200_000:
        idx = np.random.default_rng(42).choice(n, 200_000, replace=False)
        brightness, label = brightness[idx], label[idx]

    df = pd.DataFrame({"brightness": brightness, "type": label})
    chart = (
        alt.Chart(df)
        .mark_bar(opacity=0.55, binSpacing=0)
        .encode(
            alt.X("brightness:Q", bin=alt.Bin(maxbins=80), title="Mean brightness (0–255)"),
            alt.Y("count()", title="Pixel count"),
            alt.Color("type:N",
                      scale=alt.Scale(domain=["Cloud", "Land"], range=["#e45756", "#4c78a8"])),
        )
        .properties(height=220)
    )
    st.altair_chart(chart, use_container_width=True)
    st.caption("Red = cloud pixels  ·  Blue = non-cloud pixels")
```

#### New Session State Keys
None.

#### Edge Cases
- Only render inside `if mask_clouds:` — cloud mask must exist.
- 200 k subsample cap is essential for multi-megapixel images.

---

### Feature 5: Session Management (Config Save / Load)

#### Files Changed

**`app/ui/steps/step_upload.py`** — Two additions:

**1. Load config expander** (below file uploader):

```python
with st.expander("Load saved config (params.yaml)"):
    config_file = st.file_uploader("Upload a params.yaml", type=["yaml","yml"], key="config_loader")
    if config_file is not None:
        _load_params_yaml(config_file, state)
        st.success("Config loaded — parameters pre-populated for each step.")
```

**2. Save config download button** (below image preview, when any params exist):

```python
if any(k in state for k in ("upload_params","dehaze_params","enhance_params","chip_params")):
    st.download_button(
        "Download current config (params.yaml)",
        data=_build_params_yaml(state),
        file_name="params.yaml",
        mime="text/yaml",
        key="download_config",
    )
```

#### New Module-Private Functions in `app/ui/steps/step_upload.py`

```python
def _build_params_yaml(state: dict) -> bytes:
    """Serialise current *_params keys from state to UTF-8 YAML bytes."""
```

```python
def _load_params_yaml(file_like, state: dict) -> None:
    """Parse a params.yaml and write *_params keys into state (additive merge).

    Silently ignores unknown keys. Wrapped in try/except for malformed files.
    """
```

**Slider pre-population** — Each downstream step reads its stored params as widget defaults. Small defensive change required in each step:

**`app/ui/steps/step_dehaze.py`**:
```python
_dp = state.get("dehaze_params", {})
brightness_thresh = st.slider("Brightness threshold", ..., value=_dp.get("brightness_thresh", 0.75), ...)
# same for saturation_thresh, omega, t0, patch_size; use_guided via checkbox value=
```

**`app/ui/steps/step_enhance.py`**:
```python
_ep = state.get("enhance_params", {})
clip_limit = st.slider("CLAHE clip limit", ..., value=_ep.get("clip_limit", 2.0), ...)
tile_size  = st.select_slider("Tile grid size", ..., value=_ep.get("tile_size", 8))
```

**`app/ui/steps/step_chip.py`**:
```python
_cp = state.get("chip_params", {})
overlap   = st.slider("Overlap fraction", ..., value=_cp.get("overlap", 0.0), ...)
edge_mode = st.radio("Edge chip handling", ..., index=["pad","overlap"].index(_cp.get("edge_mode","pad")), ...)
naming    = st.selectbox("Chip naming", ..., index=["coords","rowcol"].index(_cp.get("naming","coords")), ...)
```

#### New Session State Keys
None permanent — `_load_params_yaml` writes into existing `*_params` keys.

#### Edge Cases
- YAML load wrapped in try/except; show `st.error(...)` on malformed file.
- Guard `_dp.get("skipped")` — if a section is `"skipped"`, don't try to read numeric values.
- Loading config does not auto-advance the step; user still clicks through.
- The `method` selectbox in step_enhance uses `index=None` by default; after load, seed it from `_ep.get("method")` — requires converting the string value to the list index.

---

### Feature 6: COCO / YOLO Annotation Export

#### New File: `app/chipping/annotation_export.py`

```python
def build_coco_manifest(
    grid: "ChipGrid",
    chip_stats: list[dict] | None = None,
    naming: str = "rowcol",
    fmt_ext: str = ".png",
    rejected_indices: list[int] | None = None,
    include_rejected: bool = False,
) -> dict:
    """Build a COCO-format dataset manifest dict.

    Produces images list (with geo_bbox metadata), empty annotations list,
    empty categories list, and info block with date and source.
    """
```

```python
def build_yolo_files(
    grid: "ChipGrid",
    chip_stats: list[dict] | None = None,
    naming: str = "rowcol",
    fmt_ext: str = ".png",
    rejected_indices: list[int] | None = None,
    include_rejected: bool = False,
) -> dict[str, bytes]:
    """Build per-chip YOLO .txt annotation files and dataset.yaml.

    Each .txt contains one full-frame annotation: 0 0.5 0.5 1.0 1.0
    (unannotated placeholder). Returns {archive_path: bytes} mapping.
    """
```

```python
def build_dataset_yaml(chip_names: list[str], image_dir="images", label_dir="labels") -> bytes:
    """Build a YOLO dataset.yaml as UTF-8 bytes."""
```

#### Files Changed

**`app/ui/steps/step_review.py`** — In Section 3 chip export controls, add:

```python
st.markdown("**Annotation export**")
export_coco = st.checkbox("COCO JSON manifest", value=False, key="export_coco_cb")
export_yolo = st.checkbox("YOLO labels + dataset.yaml", value=False, key="export_yolo_cb")
```

Update `_build_export_zip` signature to accept `export_coco` and `export_yolo` booleans. Add sections 3b and 3c:

```python
# 3b. COCO JSON
if export_coco:
    from chipping.annotation_export import build_coco_manifest
    import json
    coco = build_coco_manifest(grid, chip_stats, naming, ext, rejected_indices, include_rejected)
    zf.writestr("annotations/coco.json", json.dumps(coco, indent=2).encode())
    added = True

# 3c. YOLO
if export_yolo:
    from chipping.annotation_export import build_yolo_files
    for arc_path, content in build_yolo_files(grid, chip_stats, naming, ext, rejected_indices, include_rejected).items():
        zf.writestr(arc_path, content)
    added = True
```

#### Dependencies
- Feature 3 first (provides `chip_lat_lon_bounds` for geo_bbox).
- Import `_coords_filename` from `tile_exporter.py` to ensure filename consistency.

#### Edge Cases
- COCO `id` fields are 1-based: `chip_index + 1`.
- `geo_bbox` is non-standard COCO; document in the `info` block.
- When CRS is None: `geo_bbox = null`.
- YOLO `dataset.yaml` image path: `../chips` (chips are written there already).
- ZIP structure: COCO at `annotations/coco.json`; YOLO labels at `labels/<name>.txt`; `dataset.yaml` at root.

---

## Group C — Large

---

### Feature 7: Gray World White Balance

#### Files Changed

**`app/processing/enhancement.py`** — Add new function:

```python
def apply_gray_world(
    img_uint8: np.ndarray,
    cloud_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Apply Gray World white balance to correct atmospheric colour cast.

    Scales each RGB channel so its mean (over non-cloud pixels when
    cloud_mask is supplied) equals the grand mean of all channel means.

    Algorithm: K = mean(mean_r, mean_g, mean_b); scale_c = K / mean_c.
    Cloud pixels are excluded from mean computation but still scaled.

    Parameters
    ----------
    img_uint8 : np.ndarray
        Shape (H, W, 3), dtype uint8.
    cloud_mask : np.ndarray or None
        Shape (H, W), dtype bool. Non-cloud pixels drive mean computation.

    Returns
    -------
    np.ndarray
        Shape (H, W, 3), dtype uint8. White-balanced image.

    Notes
    -----
    Channel means near zero (< 1e-6) are left unscaled to avoid blow-out.
    Gray World assumes spectrally balanced land cover and may over-correct
    for all-desert or all-ocean scenes.
    """
    img_f = img_uint8.astype(np.float32)
    valid = ~cloud_mask if cloud_mask is not None else np.ones(img_uint8.shape[:2], bool)
    means = np.array([
        img_f[:, :, c][valid].mean() if valid.any() else img_f[:, :, c].mean()
        for c in range(3)
    ], dtype=np.float32)
    grand_mean = means.mean()
    scales = np.where(means > 1e-6, grand_mean / means, 1.0).astype(np.float32)
    return np.clip(img_f * scales[np.newaxis, np.newaxis, :], 0, 255).astype(np.uint8)
```

**`app/ui/steps/step_enhance.py`** — Add Gray World toggle before the CLAHE selector:

```python
enable_gw = st.toggle("Gray World white balance", value=False, key="enhance_gw_enable")
if enable_gw:
    st.caption("*Scales RGB channels to equalise their means, correcting colour cast from "
               "atmospheric scattering. Cloud pixels (when available) are excluded from "
               "mean computation. May over-correct on spectrally skewed scenes (all ocean, "
               "all snow).*")
```

Apply in the "Apply Enhancement" handler:

```python
src = state["dehazed_image"]
if enable_gw:
    src = apply_gray_world(src, cloud_mask=state.get("cloud_mask"))
enhanced = apply_clahe(src, ...) if method == "CLAHE" else src.copy()
```

Store in `enhance_params`:

```python
state["enhance_params"] = {
    "method": method,
    "gray_world": enable_gw,
    "clip_limit": clip_limit if method == "CLAHE" else None,
    "tile_size": tile_size if method == "CLAHE" else None,
}
```

Update stale detection to also check `state["enhance_params"].get("gray_world") != enable_gw`.

Update result caption to prefix with `"Gray World → "` when `enable_gw`.

#### New Session State Keys
None. `gray_world` flag stored inside existing `enhance_params`.

#### Edge Cases
- Stale warning must fire when `enable_gw` changes even before Apply is clicked.
- No cloud mask (dehaze skipped): `apply_gray_world(src, cloud_mask=None)` uses all pixels — graceful.
- Near-zero channel mean guard (`means > 1e-6`) prevents blow-out on near-black scenes.

---

## Implementation Order

```
Phase 1 — no dependencies (can run in parallel):
  Feature 1  Breadcrumb refactor
  Feature 7  Gray World (new function + step_enhance toggle)

Phase 2 — depends on Phase 1 structure being stable:
  Feature 2  Per-chip quality filtering   (chip_filter.py + step_chip + tile_viewer)
  Feature 4  Histogram in dehaze          (self-contained)
  Feature 5  Session management           (step_upload + defensive param-seeding in each step)

Phase 3 — depends on Feature 2's chip_stats:
  Feature 3  Chip manifest CSV            (manifest.py)
  Feature 6  COCO / YOLO export           (annotation_export.py, depends on manifest.py)
```

### Recommended Agent Assignment

| Agent | Features | Rationale |
|---|---|---|
| A | 1, 7 | Cleanest isolated changes; no shared file conflicts |
| B | 4, 5 | Both touch step_dehaze and step_upload; avoids merge conflicts |
| C | 2 | Largest single feature; owns chip_filter.py, step_chip, tile_viewer |
| D | 3, 6 | Sequential; share manifest.py infrastructure |

---

## New Files Summary

| File | Contents |
|---|---|
| `app/chipping/chip_filter.py` | `compute_chip_stats`, `apply_chip_filters` |
| `app/chipping/manifest.py` | `chip_lat_lon_bounds`, `build_manifest`, `write_manifest_csv` |
| `app/chipping/annotation_export.py` | `build_coco_manifest`, `build_yolo_files`, `build_dataset_yaml` |

## Modified Files Summary

| File | Features |
|---|---|
| `app/main.py` | F1 |
| `app/ui/steps/__init__.py` | F1 |
| `app/ui/steps/step_upload.py` | F5 |
| `app/ui/steps/step_dehaze.py` | F4, F5 |
| `app/ui/steps/step_enhance.py` | F7, F5 |
| `app/ui/steps/step_chip.py` | F2, F5 |
| `app/ui/steps/step_review.py` | F2, F3, F6 |
| `app/ui/tile_viewer.py` | F2 |
| `app/processing/enhancement.py` | F7 |
| `app/chipping/tile_exporter.py` | F2 |

---

## Future Backlog (post-Iteration 3)

- **Per-chip CLAHE at export time** — Full-image CLAHE (current) preserves spatial consistency and serves the preview, but for ML training data, applying CLAHE independently to each chip at export time produces more uniform per-chip contrast regardless of where the chip falls in the source image. The tile grid size means something very different at chip scale (e.g. 32 px tiles on a 256 px chip vs. 1250 px tiles on a 10 000 px image). Implementation: add a "Per-chip CLAHE" toggle in the chip export controls in `step_review.py`, applied in `tile_exporter.py` alongside the existing normalise option. Keep full-image CLAHE in the enhance step for preview purposes.
- **Train / val / test split assignment** — assign chips to splits in the tile viewer; export into `train/`, `val/`, `test/` subdirectories.
