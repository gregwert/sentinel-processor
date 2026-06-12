# Sentinel Processor — Iteration 5 Plan

**Prepared:** 2026-06-11
**Scope:** Structural refactors — split `_build_export_zip`, consolidate args tuple in `_export_single_chip`

---

## Overview

Two tightly related structural improvements identified during the code review:

1. `_build_export_zip` in `step_review.py` is 125+ lines and handles six distinct concerns (YAML, image, chips, manifest, COCO, YOLO). It cannot be tested independently and is hard to change safely.

2. `_export_single_chip` in `tile_exporter.py` accepts a bare 10-element positional tuple passed through `ProcessPoolExecutor.map`. Adding any new per-chip option requires manually extending every call site. `ExportConfig` already exists as a clean holder for config but is not used inside the worker.

---

## Change 1 — Split `_build_export_zip`

### Motivation

Each section of the zip (params, image, chips+manifest, annotations) is independent. Splitting makes each unit testable and keeps the top-level function as a coordinator.

### New private helpers in `step_review.py`

```python
def _add_params_section(zf: zipfile.ZipFile, state: dict) -> bool:
    """Write params.yaml to zf. Returns True if written."""

def _add_image_section(zf: zipfile.ZipFile, state: dict, img_fmt: str) -> bool:
    """Write processed_image.{png,tif} to zf. Returns True if written."""

def _add_chips_section(
    zf: zipfile.ZipFile,
    state: dict,
    chip_fmt: str,
    chip_naming: str,
    normalise_chips: bool,
    apply_ref_norm_chips: bool,
    ref_norm_method: str,
    include_rejected: bool,
    export_coco: bool,
    export_yolo: bool,
) -> bool:
    """Write chips/, manifest.csv, and optional annotation files to zf.
    Returns True if anything was written."""
```

### Revised `_build_export_zip`

Becomes a thin coordinator: creates the ZipFile, calls the three helpers, returns bytes or None.

```python
def _build_export_zip(...) -> bytes | None:
    buf = io.BytesIO()
    added = False
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        if export_params:
            added |= _add_params_section(zf, state)
        if export_image:
            added |= _add_image_section(zf, state, img_fmt)
        if export_chips:
            added |= _add_chips_section(zf, state, ...)
    return buf.getvalue() if added else None
```

### No behaviour changes

All logic moves verbatim — the split is mechanical. No new features, no new state keys.

---

## Change 2 — Replace the 10-element tuple with a worker dataclass

### Motivation

`_export_single_chip` currently unpacks:

```python
chip_array, chip_meta, out_path, fmt, normalise, global_mean, global_std, \
apply_ref_norm, reference_stats, ref_norm_method = args
```

Positional unpacking is fragile — a missing element silently shifts every downstream variable. `ExportConfig` already holds all the config fields; the only positional args are the per-chip data (`chip_array`, `chip_meta`, `out_path`).

### New worker input type

```python
@dataclass
class _ChipTask:
    """Single-chip export task passed to the process pool worker."""
    chip_array: np.ndarray
    chip_meta: dict
    out_path: str
    config: ExportConfig
```

`_ChipTask` is defined in `tile_exporter.py` alongside `ExportConfig`. It is a plain dataclass — picklable by default.

### Changes in `export_chips`

Replace the args tuple construction:

```python
# Before
args_list.append((chip_array, chip_meta, out_path, fmt, normalise,
                  global_mean, global_std, _apply_ref_norm,
                  _reference_stats, _ref_norm_method))

# After
args_list.append(_ChipTask(
    chip_array=chip_array,
    chip_meta=chip_meta,
    out_path=out_path,
    config=config,
))
```

### Changes in `_export_single_chip`

```python
def _export_single_chip(task: _ChipTask) -> str:
    chip_array = task.chip_array
    chip_meta  = task.chip_meta
    out_path   = task.out_path
    cfg        = task.config
    # use cfg.fmt, cfg.normalise, cfg.global_stats, cfg.apply_ref_norm, etc.
```

### Backward compatibility

`export_chips` already resolves individual kwargs into a config object early in the function. No external call sites need updating — callers that pass individual kwargs or an `ExportConfig` continue to work unchanged.

---

## Implementation Order

```
Phase 1 (independent):
  tile_exporter.py   — add _ChipTask, update _export_single_chip and export_chips

Phase 2 (depends on Phase 1 being stable):
  step_review.py     — split _build_export_zip into three helpers
```

Both phases can be reviewed independently before merging.

---

## Testing

- Existing `export_chips` integration path (called from step_review) is the primary test.
- Add a unit test for `_add_chips_section` that mocks `export_chips` and verifies zip contents.
- Verify the split does not change the exported zip structure by comparing a reference zip before and after.

---

## Non-goals

- No new features.
- No changes to ExportConfig fields.
- No changes to the UI or state keys.
