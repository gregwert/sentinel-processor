# Sentinel-2 Image Processor

A browser-based wizard for processing 16-bit Sentinel-2 L2A satellite imagery. Load a GeoTIFF, remove atmospheric haze, enhance local contrast, chip the image into tiles, then export everything as a single ZIP.

---

## Features

- **Upload** — load a Sentinel-2 L2A multi-band GeoTIFF; select R/G/B bands and percentile-stretch to uint8
- **Dehaze** — Dark Channel Prior (DCP) dehazing with cloud-adaptive atmospheric light estimation
- **Enhance** — CLAHE contrast enhancement in LAB colour space
- **Chip** — grid-based tiling with configurable overlap, edge handling (pad or clamp), and a live grid overlay preview
- **Review & Export** — download processing parameters (YAML), the processed image (PNG or GeoTIFF), and all chips in a single ZIP

---

## Project Layout

```
sentinel-processor/
├── app/
│   ├── main.py                  # Streamlit entry point / step router
│   ├── chipping/
│   │   ├── gdal_chipper.py      # ChipGrid and window computation
│   │   └── tile_exporter.py     # ExportConfig and chip writing
│   ├── processing/
│   │   ├── dehazing.py          # Dehazer dataclass (DCP pipeline)
│   │   ├── enhancement.py       # CLAHE wrapper
│   │   ├── pipeline.py          # Batch pipeline helper
│   │   └── preprocess.py        # Band selection and percentile stretch
│   └── ui/
│       ├── cloud_overlay.py     # Cloud mask visualisation
│       ├── grid_overlay.py      # Chip grid preview renderer
│       ├── tile_viewer.py       # Paginated chip tile viewer
│       └── steps/               # One module per wizard step
├── tests/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt             # Python dependencies
├── environment.yml              # Conda environment (optional)
├── launch.py                    # Local Windows launcher (SSL patch)
└── run.bat                      # Double-click launcher for Windows
```

---

## Local Setup (Windows)

### Prerequisites

- Python 3.11 ([python.org](https://www.python.org/downloads/))
- Git (to clone the repo)

> **Note on GDAL:** the `rasterio` Windows wheel bundles GDAL internally — no separate GDAL system install is required.

### Steps

```bat
git clone <repo-url>
cd sentinel-processor
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Running

```bat
python launch.py
```

Or double-click `run.bat`.

The app opens at **http://localhost:8501**.

`launch.py` patches Python's SSL certificate loader before starting Streamlit. This works around an intermittent Windows issue where malformed certificates in the OS certificate store cause Tornado (Streamlit's HTTP server) to crash on startup.

### Optional: Conda

```bat
conda create -n sentinel-processor python=3.11 -y
conda activate sentinel-processor
pip install -r requirements.txt
python launch.py
```

---

## Docker Deployment

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (or Docker Engine + Compose plugin)

### Build and run

```bash
docker compose up --build
```

The app will be available at **http://localhost:8501**.

On subsequent starts (no code changes):

```bash
docker compose up
```

To stop:

```bash
docker compose down
```

### What the container does

| Setting | Value |
|---|---|
| Base image | `osgeo/gdal:ubuntu-small-3.8.4` |
| Exposed port | `8501` |
| Max upload size | `2048 MB` |
| Restart policy | `unless-stopped` |
| Health check | `GET /stcore/health` every 30 s |

The Compose file mounts two paths:

| Mount | Purpose |
|---|---|
| `./sample_data` → `/app/sample_data` (read-only) | Optional sample GeoTIFFs — place test files here to access them via the upload step |
| Named volume `chip_output` → `/tmp/chips` | Scratch space used during chip export |

Create the sample data directory if you want to use it:

```bash
mkdir sample_data
```

### Building the image without Compose

```bash
docker build -t sentinel-processor .
docker run -p 8501:8501 sentinel-processor
```

---

## Using the App

1. **Upload** — drag in a Sentinel-2 GeoTIFF. Select which bands map to R, G, B (defaults: 4, 3, 2 for TCI composites). Adjust the percentile stretch if the image looks washed out or clipped.

2. **Dehaze** — enable DCP dehazing and tune parameters, or skip to pass the stretched image through unchanged. Enable "Cloud-adaptive atmospheric light" to prevent bright clouds from biasing the haze estimate.

3. **Enhance** — apply CLAHE to boost local contrast, or select "None". CLAHE is run only on the L channel in LAB space to avoid colour shifts.

4. **Chip** — set chip dimensions (pixels or metres), overlap fraction, and edge handling:
   - **Pad** — edge chips that don't divide evenly are zero-padded to the full chip size.
   - **Overlap** — edge chips are shifted inward so every chip is full size with no black borders.
   
   The grid preview shows solid yellow lines for the no-overlap reference grid and dashed orange lines for any additional chip boundaries introduced by overlap or edge clamping.

5. **Review & Export** — check which outputs to include (parameters YAML, processed image, chips), preview them in-page, then click **Export Selected** to download a ZIP.

---

## Input Requirements

| Property | Requirement |
|---|---|
| Format | GeoTIFF (`.tif` / `.tiff`) |
| Bit depth | 16-bit (Sentinel-2 L2A standard) |
| Bands | At least 3 (select R/G/B during upload) |
| CRS | Any projected or geographic CRS; metres-to-pixels conversion uses the embedded geotransform |
| Max file size | 2048 MB (configurable in `launch.py` or `docker-compose.yml`) |

---

## Configuration

All runtime settings are hardcoded in `launch.py` (local) or `docker-compose.yml` (Docker). The most commonly adjusted values:

| Setting | Where | Default |
|---|---|---|
| Upload size limit | `launch.py` `sys.argv` / `docker-compose.yml` env | `2048` MB |
| Streamlit port | `launch.py` `sys.argv` / Dockerfile `ENTRYPOINT` | `8501` |

---

## Running Tests

```bat
python -m pytest tests/
```

Tests cover chipping geometry, DCP dehazing, CLAHE enhancement, and the percentile stretch preprocessing step.
