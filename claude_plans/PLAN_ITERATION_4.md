# Sentinel Processor — Iteration 4 Plan

**Prepared:** 2026-06-11
**Scope:** Reference-based radiometric normalisation

---

## Overview

Adds support for uploading cloudless reference images and using their per-band statistics to anchor the radiometry of target images. Ensures chips from different acquisition dates are radiometrically consistent — critical for ML training datasets.

Two methods supported:
- **Histogram matching** — shifts full per-band histogram to match reference; handles non-linear illumination differences
- **Linear (mean/std)** — per-band linear rescaling; simpler, more robust when averaging across multiple references

Can be applied to the full image in the enhance step, per-chip at export time, or both independently.

---

## Implementation Order

```
Phase 1 (no dependencies — run in parallel):
  processing/reference_norm.py   (new — all other changes import from here)
  step_upload.py                 (reference uploader — no imports from reference_norm)

Phase 2 (depends on Phase 1):
  step_enhance.py                (imports apply_reference_normalisation)
  tile_exporter.py               (imports apply_reference_normalisation in worker)

Phase 3 (depends on Phase 2):
  step_review.py                 (wires ExportConfig new fields)

Tests: tests/test_reference_norm.py (can run in parallel with Phase 1)
```

---

## New File: `app/processing/reference_norm.py`

### `compute_reference_stats`

```python
def compute_reference_stats(images: List[np.ndarray]) -> dict:
```

**Parameters:** non-empty list of uint8 (H, W, 3) arrays — already percentile-stretched using the same band/stretch settings as the target.

**Returns dict:**
- `"n"` — int, number of reference images
- `"cdfs"` — list of 3 np.ndarray shape (256,) float32, per-band normalised CDFs averaged across all references
- `"mean"` — np.ndarray shape (3,) float32, per-band mean averaged across references
- `"std"` — np.ndarray shape (3,) float32, per-band std averaged across references

**Implementation (pure numpy, no skimage — required for picklability in ProcessPoolExecutor):**
```
For each image, for each channel c:
    hist, _ = np.histogram(image[:,:,c].flatten(), bins=256, range=(0, 256))
    cdf = hist.cumsum().astype(np.float32)
    cdf /= (cdf[-1] + 1e-6)
    accumulate cdfs, means, stds
Average accumulated values across all images.
```

Storage: 3 × 256 × 4 bytes = 3 KB — safe for session state.

**Edge cases:** empty list → raise ValueError.

---

### `apply_reference_normalisation`

```python
def apply_reference_normalisation(
    img_uint8: np.ndarray,
    reference_stats: dict,
    method: str,
) -> np.ndarray:
```

**Histogram matching (`method == "histogram"`):**
```
For each channel c:
    Compute target CDF from img_uint8[:,:,c]
    Build 256-entry LUT:
        lut[v] = np.searchsorted(reference_stats["cdfs"][c], target_cdf[v])
    Apply: result[:,:,c] = lut[img_uint8[:,:,c]]
Clip to [0,255], cast uint8.
```

**Linear matching (`method == "linear"`):**
```
For each channel c:
    src_mean = img_uint8[:,:,c].mean()
    src_std  = img_uint8[:,:,c].std() + 1e-6
    ref_mean = reference_stats["mean"][c]
    ref_std  = reference_stats["std"][c]
    if ref_std < 1e-6: leave channel unchanged
    else: result = (src - src_mean) / src_std * ref_std + ref_mean
Clip to [0,255], cast uint8.
```

Raise ValueError for unknown method.

---

## Modified: `app/ui/steps/step_upload.py`

### New import
```python
from processing.reference_norm import compute_reference_stats
```

### Cache key design
```python
_ref_files_key = str(sorted([f.file_id for f in ref_files]))
_ref_params_fragment = str((band_r, band_g, band_b, p_low, p_high, per_band))
_ref_stats_key = str((_ref_files_key, _ref_params_fragment))
```

Compare against `state.get("_ref_stats_key")` — recompute on mismatch.

### Cache invalidation on band/stretch change
Inside the existing `if state.get("_upload_params_key") != params_key or "stretched_image" not in state:` block, add:
```python
state.pop("reference_stats", None)
state.pop("_ref_stats_key", None)
```

### UI placement
After the metadata caption (`st.caption(f"{img.shape[1]}×{img.shape[0]} px ...")`) and before the download-config button.

```python
st.divider()
st.subheader("Reference Images (optional)")
st.caption("*Upload cloudless reference TIFFs of the same area. Statistics are averaged across all references and used in Step 3 to anchor the target image's radiometry.*")

ref_files = st.file_uploader(
    "Upload reference Sentinel-2 TIFFs",
    type=["tif", "tiff"],
    accept_multiple_files=True,
    key="ref_uploader",
)
```

### Processing logic
```python
if ref_files:
    # compute _ref_stats_key
    if state.get("_ref_stats_key") != _ref_stats_key:
        # read each reference with same band_indices, p_low, p_high, per_band
        # collect stretched uint8 arrays
        # call compute_reference_stats(stretched_refs)
        # store state["reference_stats"] and state["_ref_stats_key"]
    # show summary: "N reference images · mean R/G/B: x / y / z"
else:
    state.pop("reference_stats", None)
    state.pop("_ref_stats_key", None)
```

Each reference file is read into a temp file → `read_sentinel_tiff` → `percentile_stretch` (same args as target). Errors per file are shown via `st.error` and that file is skipped; processing continues with remaining references.

### New session state keys
| Key | Written by | Cleared by |
|---|---|---|
| `reference_stats` | step_upload | step_upload (no files / param change) |
| `_ref_stats_key` | step_upload | step_upload (no files / param change) |

---

## Modified: `app/ui/steps/step_enhance.py`

### New import
```python
from processing.reference_norm import apply_reference_normalisation
```

### New controls — placement
After the Gray World toggle+caption, before the CLAHE method selectbox.

```python
_has_ref_stats = "reference_stats" in state
enable_ref_norm = st.toggle(
    "Reference normalisation",
    value=bool(_ep.get("ref_norm", False)) and _has_ref_stats,
    disabled=not _has_ref_stats,
    key="enhance_ref_norm_enable",
)
if not _has_ref_stats:
    st.caption("*Upload reference images in Step 1 to enable. Anchors the target image's radiometry to cloudless reference acquisitions.*")
elif enable_ref_norm:
    ref_norm_method = st.radio(
        "Normalisation method",
        ["Histogram matching", "Linear (mean/std)"],
        index=...,  # seed from _ep.get("ref_norm_method", "Histogram matching")
        horizontal=True, key="enhance_ref_norm_method",
    )
    st.caption("*Histogram matching shifts the full per-band histogram — handles non-linear differences. Linear rescaling is simpler and more robust across multiple references. When Gray World is also enabled it runs first; the two are somewhat redundant — reference norm alone is usually sufficient.*")
else:
    ref_norm_method = _ep.get("ref_norm_method", "Histogram matching")
```

### Stale detection additions
```python
or state["enhance_params"].get("ref_norm") != enable_ref_norm
or state["enhance_params"].get("ref_norm_method") != ref_norm_method
```

### Method-change clear additions
```python
or state.get("enhance_params", {}).get("ref_norm") != enable_ref_norm
or state.get("enhance_params", {}).get("ref_norm_method") != ref_norm_method
```

### Apply Enhancement handler — processing order
```python
src = state["dehazed_image"]
if enable_gw:
    src = apply_gray_world(src, cloud_mask=state.get("cloud_mask"))
if enable_ref_norm and "reference_stats" in state:
    method_key = "histogram" if ref_norm_method == "Histogram matching" else "linear"
    src = apply_reference_normalisation(src, state["reference_stats"], method_key)
if method == "CLAHE":
    enhanced = apply_clahe(src, clip_limit, (tile_size, tile_size))
else:
    enhanced = src.copy()
```

### enhance_params additions
```python
"ref_norm": enable_ref_norm,
"ref_norm_method": ref_norm_method,
```

### Result caption
```python
gw_prefix  = "Gray World → " if p.get("gray_world") else ""
ref_prefix = "Ref Norm → "   if p.get("ref_norm")   else ""
# prefix CLAHE or "No enhancement" caption with gw_prefix + ref_prefix
```

---

## Modified: `app/chipping/tile_exporter.py`

### `ExportConfig` — new fields
```python
apply_ref_norm: bool = False
reference_stats: dict = None
ref_norm_method: str = "histogram"
```

### `_export_single_chip` — tuple extension
**Old:** `(chip_array, chip_meta, out_path, fmt, normalise, global_mean, global_std)`
**New:** `(chip_array, chip_meta, out_path, fmt, normalise, global_mean, global_std, apply_ref_norm, reference_stats, ref_norm_method)`

Add after existing z-score block, before format dispatch:
```python
if apply_ref_norm and reference_stats is not None:
    from processing.reference_norm import apply_reference_normalisation
    chip_array = apply_reference_normalisation(chip_array, reference_stats, ref_norm_method)
```

### `export_chips` — args_list construction
Extract new config fields with `getattr(..., default)` for backward compatibility. Append new fields to the args tuple.

---

## Modified: `app/ui/steps/step_review.py`

### `_build_export_zip` — new parameters
```python
apply_ref_norm_chips: bool = False,
ref_norm_method: str = "histogram",
```
(placed before `include_rejected`)

### Chips section — UI controls
Inside `if export_chips_cb:` block, after normalise_chips checkbox:
```python
apply_ref_norm_chips = False
ref_norm_method_export = "histogram"
if state.get("reference_stats"):
    apply_ref_norm_chips = st.checkbox("Apply reference normalisation per chip", value=False, key="export_ref_norm_cb")
    if apply_ref_norm_chips:
        _rn_choice = st.radio("Method", ["Histogram matching", "Linear (mean/std)"], horizontal=True, key="export_ref_norm_method")
        ref_norm_method_export = "histogram" if _rn_choice == "Histogram matching" else "linear"
        st.caption("*Applied per-chip using pre-computed reference statistics from Step 1. Independent of any reference normalisation applied to the full image in Step 3.*")
```

Initialise both to `False` / `"histogram"` in the chip-skipped else branch.

### ExportConfig construction in `_build_export_zip`
```python
config = ExportConfig(
    fmt=chip_fmt, naming=chip_naming,
    normalise=normalise_chips, global_stats=state.get("global_stats"),
    apply_ref_norm=apply_ref_norm_chips,
    reference_stats=state.get("reference_stats") if apply_ref_norm_chips else None,
    ref_norm_method=ref_norm_method,
)
```

Update `_build_export_zip` call in `render` to pass `apply_ref_norm_chips` and `ref_norm_method_export`.

---

## Tests: `tests/test_reference_norm.py`

1. `test_compute_single_image` — n==1, cdfs is list of 3 arrays shape (256,), values in [0,1], mean/std shape (3,) float32
2. `test_compute_multiple_images` — averaged CDFs/means between individual image values
3. `test_compute_empty_raises` — ValueError on empty list
4. `test_cdfs_monotonic` — each CDF is non-decreasing
5. `test_apply_histogram_dtype` — output uint8
6. `test_apply_histogram_shape` — shape unchanged
7. `test_apply_histogram_range` — values in [0,255]
8. `test_apply_histogram_shifts_mean` — target mean 100, ref mean 200 → output mean closer to 200
9. `test_apply_linear_dtype` — output uint8
10. `test_apply_linear_range` — values in [0,255]
11. `test_apply_linear_shifts_statistics` — output stats closer to reference
12. `test_apply_invalid_method_raises` — ValueError on unknown method
13. `test_apply_linear_degenerate_reference` — near-zero ref std → channel unchanged
14. `test_picklable` — `pickle.dumps(compute_reference_stats([img]))` succeeds

---

## Edge Cases

- **Reference image size mismatch:** Fine — only per-band histograms/stats are stored, no spatial alignment required
- **Band count mismatch:** `read_sentinel_tiff` raises; caught per-file, displayed via `st.error`, that file skipped
- **params.yaml round-trip:** `reference_stats` is not in YAML (binary data). `ref_norm`/`ref_norm_method` are in `enhance_params` YAML. On reload, toggle pre-checked but disabled until user re-uploads reference images
- **Both full-image and per-chip ref norm enabled:** Independent. Per-chip operates on already-normalised chip data — valid and requires no special handling
- **NPY export + ref norm:** Ref norm runs after z-score (if also enabled), before `np.save`. Output is uint8 reference-normalised array

---

## Future Backlog (post-Iteration 4)

- Per-chip CLAHE at export time — apply CLAHE independently to each chip for more uniform contrast across ML training chips (see notes in PLAN_ITERATION_3.md)
- Train / val / test split assignment in tile viewer
