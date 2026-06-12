# Sentinel-2 Image Processor

A browser-based wizard for processing 16-bit Sentinel-2 L2A satellite imagery into ML-ready chip datasets. Load a GeoTIFF, remove atmospheric haze, apply radiometric normalisation and contrast enhancement, chip the image into tiles with quality filtering, then export everything — chips, manifest, and annotations — as a single ZIP.

### Features

- **Upload** — load a Sentinel-2 L2A multi-band GeoTIFF; select R/G/B bands and percentile-stretch to uint8; optionally upload cloudless reference images for radiometric normalisation
- **Dehaze** — Dark Channel Prior (DCP) dehazing with cloud-adaptive atmospheric light estimation; brightness histogram shows before/after distribution
- **Enhance** — optional Gray World white balance, reference-based radiometric normalisation (histogram matching or linear mean/std), and CLAHE contrast enhancement
- **Chip** — grid-based tiling with configurable overlap, edge handling (pad or overlap-shift), and a live grid overlay preview
- **Review & Export** — download processing parameters, the processed image, and all chips in a single ZIP

---

## Project Layout

```
sentinel-processor/
├── app/
│   ├── main.py                      # Streamlit entry point and step router
│   ├── utils.py                     # Shared utilities (YAML safe serialisation)
│   ├── chipping/
│   │   ├── gdal_chipper.py          # ChipGrid dataclass and window computation
│   │   ├── tile_exporter.py         # ExportConfig, _ChipTask, parallel chip export
│   │   ├── chip_filter.py           # Per-chip cloud and variance quality filters
│   │   ├── manifest.py              # Manifest CSV builder with geographic bounds
│   │   └── annotation_export.py     # COCO JSON and YOLO label export
│   ├── processing/
│   │   ├── dehazing.py              # Dehazer dataclass (Dark Channel Prior pipeline)
│   │   ├── enhancement.py           # CLAHE and Gray World white balance
│   │   ├── reference_norm.py        # Reference-based radiometric normalisation
│   │   ├── pipeline.py              # Batch pipeline helper
│   │   └── preprocess.py            # Band selection and percentile stretch
│   └── ui/
│       ├── cloud_overlay.py         # Cloud mask visualisation
│       ├── grid_overlay.py          # Chip grid preview renderer
│       ├── preview.py               # Stage-by-stage image comparison
│       ├── tile_viewer.py           # Paginated chip tile viewer
│       └── steps/                   # One module per wizard step
│           ├── __init__.py          # Breadcrumb and step navigation widgets
│           ├── step_upload.py
│           ├── step_dehaze.py
│           ├── step_enhance.py
│           ├── step_chip.py
│           └── step_review.py
├── tests/
│   ├── test_preprocess.py
│   ├── test_dehazing.py
│   ├── test_enhancement.py
│   ├── test_chipping.py
│   ├── test_reference_norm.py
│   ├── test_chip_filter.py
│   ├── test_manifest.py
│   ├── test_annotation_export.py
│   ├── test_pipeline.py
│   └── test_utils.py
├── claude_plans/                    # Implementation plans (for reference)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt                 # Python dependencies
├── environment.yml                  # Conda environment (optional)
├── launch.py                        # Local Windows launcher (SSL patch)
└── run.bat                          # Double-click launcher for Windows
```

---

## Deployment

This app can be deployed via docker or locally with python.

Once deployed the app is available at [**http://localhost:8501**](http://localhost:8501).

### Docker

#### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (or Docker Engine + Compose plugin)

#### Build and run

```bash
docker compose up --build
```

On subsequent starts (no code changes):

```bash
docker compose up
```

To stop:

```bash
docker compose down
```

### Local (Python)

#### Prerequisites

- Python 3.11 ([python.org](https://www.python.org/downloads/))

> **Note on GDAL:** the `rasterio` Windows wheel bundles GDAL internally — no separate GDAL system install is required.

#### Setup

##### venv

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

##### (ALT) Conda

```bat
conda create -n sentinel-processor python=3.11 -y
conda activate sentinel-processor
pip install -r requirements.txt
```

#### Run

```bat
python launch.py
```
or
```bat
run.bat
```
---

## Using the App

### Step 1 — Upload

Drag in a Sentinel-2 GeoTIFF. Select which bands map to R, G, B (defaults: 1, 2, 3). Adjust the low and high percentile clip sliders if the image looks washed out or clipped.

**Reference images (optional):** upload one or more cloudless acquisitions of the same area to compute reference radiometric statistics. These are used in Step 3 to anchor the target image's band distributions. The same band selection and stretch settings are applied to all reference images, so statistics are comparable.

**Session config:** if you have a `params.yaml` from a previous export, load it via the expander at the top of the page to pre-populate all wizard controls. A download link for the current config is shown once an image is loaded.

---

### Step 2 — Dehaze

Enable Dark Channel Prior (DCP) dehazing and tune the parameters, or skip to pass the stretched image through unchanged.

- **Cloud-adaptive atmospheric light** — prevents bright cloud pixels from biasing the haze estimate; estimated from the darkest pixels in the dark channel, excluding cloud regions
- **Brightness histogram** — shows the pixel value distribution before and after dehazing to help judge the effect

Key parameters:

| Parameter | Effect |
|---|---|
| Patch size | Size of the local minimum window for dark channel estimation; larger = smoother |
| Omega (haze amount) | Fraction of haze to remove; 1.0 = full removal, lower = gentler |
| Transmission floor | Minimum transmission to preserve; prevents very dark regions from being over-amplified |
| Guided filter radius | Smoothing radius for the transmission map refinement |

---

### Step 3 — Enhance

All enhancements are optional and can be combined:

- **Gray World white balance** — scales RGB channels so their per-channel means match the scene mean, correcting colour cast from atmospheric scattering. Cloud pixels are excluded from the mean computation. Works best on spectrally diverse scenes; may over-correct on all-ocean or all-desert imagery.

- **Reference normalisation** — available when reference images were uploaded in Step 1. Anchors the target image's radiometry to match the reference statistics:
  - *Histogram matching* — shifts the full per-band histogram; handles non-linear illumination differences
  - *Linear (mean/std)* — per-band linear rescaling; simpler and more robust when averaging across multiple references

- **CLAHE** — Contrast Limited Adaptive Histogram Equalisation applied to the L channel in LAB colour space to boost local contrast without hue distortion. Tune the clip limit and tile grid size.

Select **None** to skip enhancement entirely and pass the dehazed image through unchanged.

---

### Step 4 — Chip

Set chip dimensions (pixels), overlap fraction, and edge handling:

- **Pad** — chips at the right and bottom edges that don't fill the chip size exactly are zero-padded. Every chip covers a unique area; no pixels are counted twice.
- **Overlap** — edge chips are shifted inward so every chip is exactly the requested size with no black borders. Edge chips overlap their neighbours slightly.

The **grid preview overlay** shows the chip boundaries on the processed image:
- Solid lines mark the regular no-overlap grid
- Dashed lines mark any boundary that deviates from the regular grid (overlap-shifted edges)

**Quality filtering (optional):**

| Filter | Description |
|---|---|
| Cloud coverage | Reject chips where the fraction of cloud pixels exceeds the threshold |
| Variance | Reject chips with variance below the threshold (featureless / uniform regions) |

Rejected chips are shown with a red tint in the chip viewer. You can toggle whether rejected chips are displayed and choose to include them in the export.

---

### Step 5 — Review & Export

Preview all processing parameters and the processed image, then select what to include in the export ZIP.

**Processed image:** PNG or GeoTIFF (with original CRS and geotransform).

**Chips:**
- Format: PNG, JPEG, GeoTIFF, or NumPy `.npy`
- Naming: row/column index or geographic coordinate origin
- *Z-score normalisation* — normalise each chip using global mean/std computed from the full enhanced image
- *Reference normalisation* — apply the reference statistics from Step 1 independently to each chip (useful for ML training consistency across dates)

**Chip manifest CSV** — written alongside every chip export. Columns:

| Column | Description |
|---|---|
| chip_index | Sequential index (row-major) |
| row, col | Grid position |
| pixel_x_min/max, pixel_y_min/max | Pixel bounding box in source image |
| lon_min/max, lat_min/max | WGS-84 geographic bounds (empty if CRS unknown) |
| cloud_pct | Cloud pixel fraction (0–1) |
| variance | Pixel variance |
| filename | Chip filename as exported |
| rejected | true / false |

**Annotation export:**
- *COCO JSON* — chip image entries with `geo_bbox` geographic bounding box extension; annotations are empty (suitable as a dataset manifest for labelling)
- *YOLO labels* — one empty `.txt` per chip plus `dataset.yaml`; add bounding boxes to the `.txt` files before training

Click **Export Selected** to generate and download `export.zip`.

---

## Input Requirements

| Property | Requirement |
|---|---|
| Format | GeoTIFF (`.tif` / `.tiff`) |
| Bit depth | 16-bit (Sentinel-2 L2A standard) |
| Bands | At least 3 (select R/G/B band indices during upload) |
| CRS | Any projected or geographic CRS supported by rasterio |
| Max file size | 2048 MB (configurable in `launch.py` or `docker-compose.yml`) |

Reference images must be GeoTIFFs from the same sensor and area with the same or greater band count as the target image. Spatial resolution and extent do not need to match — only per-band histograms and statistics are used.

---

## Configuration

| Setting | Where | Default |
|---|---|---|
| Upload size limit | `launch.py` / `docker-compose.yml` | `2048` MB |
| Streamlit port | `launch.py` / Dockerfile `ENTRYPOINT` | `8501` |

---

## Running Tests

```bat
python -m pytest tests/
```

Test coverage:

| Module | What is tested |
|---|---|
| `test_preprocess.py` | Percentile stretch output properties, flat-band edge case, per-band vs global mode, TIFF read and metadata |
| `test_dehazing.py` | DCP end-to-end, dark channel, transmission map, guided filter refinement, radiance recovery, atmospheric light estimation, cloud detection and pixel restoration |
| `test_enhancement.py` | CLAHE contrast increase, Gray World channel balancing |
| `test_chipping.py` | Grid geometry, chip affine transforms, overlap/pad edge modes, coordinate-based chip naming (geographic and projected CRS), chip export formats (PNG, JPEG, GeoTIFF, NPY), rejection filtering |
| `test_reference_norm.py` | Single and multi-image stats, CDF monotonicity, histogram matching, linear normalisation, degenerate reference band, error cases, picklability |
| `test_chip_filter.py` | Chip stats (cloud fraction, variance, edge chip denominator), cloud and variance filter combinations, partition correctness |
| `test_manifest.py` | Geographic bounds computation, pixel bounding boxes, manifest row fields and rowcol naming, CSV header and round-trip |
| `test_annotation_export.py` | COCO JSON structure (image entries, geo_bbox, 1-based IDs, rowcol naming), YOLO label files and dataset.yaml, rejection handling |
| `test_pipeline.py` | End-to-end pipeline output, no-dehaze mode, error cases (invalid band, missing file) |
| `test_utils.py` | `_yaml_safe` tuple-to-list conversion (top-level, nested, scalars, YAML round-trip) |
