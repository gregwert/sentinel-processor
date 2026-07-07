# Sentinel-2 Image Processor

A browser-based pipeline for processing 16-bit Sentinel-2 L2A satellite imagery into ML-ready chip datasets. Upload a GeoTIFF, remove atmospheric haze, apply radiometric normalisation and contrast enhancement, chip the image into tiles with quality filtering, then export chips, a manifest CSV, and optional annotations as a ZIP archive.

## Architecture

The project is split into two independent services:

**Backend** (`backend/`) — FastAPI service (`sentinel_backend` package) that receives GeoTIFF uploads, runs all image processing (stretch, cloud mask, DCP dehaze, CLAHE enhance), computes chip grids, and assembles export archives. Stores session state as `.npy` arrays on disk identified by UUID session IDs. Runs on port 8000.

**Frontend** (`frontend/`) — Streamlit service (`sentinel_frontend` package) that provides a 5-step wizard UI. Contains no processing logic; all computation is delegated to the backend via `api_client.py`. Runs on port 8501.

Both services are defined in `docker-compose.yml` at the repo root. The frontend waits for the backend healthcheck before starting.

## Quick Start

Requires [Docker Desktop](https://www.docker.com/products/docker-desktop/) (or Docker Engine + Compose plugin).

```bash
docker compose up --build
```

Then open http://localhost:8501.

Subsequent starts (no code changes):

```bash
docker compose up
```

Stop:

```bash
docker compose down
```

## Services

| Service  | URL                              | Notes                          |
|----------|----------------------------------|--------------------------------|
| Frontend | http://localhost:8501            | Streamlit wizard UI            |
| Backend  | http://localhost:8000            | FastAPI REST API               |
| API docs | http://localhost:8000/docs       | Interactive OpenAPI docs       |
| Health   | http://localhost:8000/healthz    | Used by Docker healthcheck     |

## Development Setup

### Backend

```bash
cd backend
pip install -e .
uvicorn sentinel_backend.api.main:app --reload
```

### Frontend

```bash
cd frontend
pip install -r requirements.txt
BACKEND_URL=http://localhost:8000 python -m streamlit run sentinel_frontend/main.py
```

## Running Tests

```bash
# Backend
cd backend && pytest

# Frontend
cd frontend && pytest
```

## Project Layout

```
sentinel-processor/
├── backend/
│   ├── Dockerfile
│   ├── pyproject.toml
│   ├── sentinel_backend/
│   │   ├── api/
│   │   │   ├── main.py              # FastAPI app and router registration
│   │   │   ├── deps.py              # Shared dependencies (session resolution)
│   │   │   └── routers/
│   │   │       ├── sessions.py      # Session lifecycle and file upload
│   │   │       ├── processing.py    # Stretch, cloud mask, dehaze, enhance, preview
│   │   │       ├── chips.py         # Chip grid, filters, thumbnail, manifest
│   │   │       └── export.py        # ZIP export job and download
│   │   ├── processing/              # Image processing algorithms
│   │   ├── chipping/                # Grid computation, export, manifest, annotations
│   │   ├── storage.py               # Session storage abstraction
│   │   ├── jobs.py                  # Background job tracking
│   │   ├── models.py                # Pydantic v2 request/response models
│   │   └── utils.py
│   └── tests/
├── frontend/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── sentinel_frontend/
│   │   ├── main.py                  # Streamlit entry point and step router
│   │   ├── api_client.py            # All backend API calls
│   │   ├── models.py                # Shared data models
│   │   └── ui/
│   │       ├── steps/               # One module per wizard step
│   │       ├── cloud_overlay.py
│   │       └── grid_overlay.py
│   └── tests/
└── docker-compose.yml
```
