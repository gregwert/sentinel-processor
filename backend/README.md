# sentinel-processor — Backend

FastAPI service that handles all image processing for the sentinel-processor pipeline. It accepts Sentinel-2 GeoTIFF uploads, runs sequential processing jobs (percentile stretch, cloud masking, Dark Channel Prior dehazing, Gray World white balance, reference normalisation, CLAHE enhancement), computes chip grids, applies quality filters, and assembles export ZIP archives. Session state is stored as `.npy` arrays on disk under `DATA_DIR`, keyed by UUID session IDs.

## API Overview

All routes are under the `/v1` prefix. Processing steps that are computationally expensive (`/stretch`, `/dehaze`, `/enhance`, `/export`) start background jobs and return a `job_id`; poll `GET /v1/jobs/{job_id}` for status.

### Sessions

| Method   | Path                                    | Description                                   |
|----------|-----------------------------------------|-----------------------------------------------|
| `POST`   | `/v1/sessions`                          | Create a new session                          |
| `GET`    | `/v1/sessions/{id}`                     | Get session metadata and stage readiness      |
| `DELETE` | `/v1/sessions/{id}`                     | Delete session and all artifacts              |
| `POST`   | `/v1/sessions/{id}/source`              | Upload source GeoTIFF                        |
| `POST`   | `/v1/sessions/{id}/references`          | Upload one or more reference GeoTIFFs        |

### Processing

| Method   | Path                                    | Description                                   |
|----------|-----------------------------------------|-----------------------------------------------|
| `POST`   | `/v1/sessions/{id}/stretch`             | Band selection and percentile stretch (async) |
| `POST`   | `/v1/sessions/{id}/cloud-mask`          | Detect clouds and save binary mask            |
| `POST`   | `/v1/sessions/{id}/dehaze`              | DCP dehazing (async)                         |
| `POST`   | `/v1/sessions/{id}/enhance`             | Gray World / ref norm / CLAHE (async)        |
| `GET`    | `/v1/sessions/{id}/preview/{stage}`     | PNG preview of a processing stage            |
| `GET`    | `/v1/sessions/{id}/histogram`           | Brightness histogram; `?stage=` query param  |

Valid stage values: `stretched`, `dehazed`, `enhanced`, `cloud_mask`.

### Chips

| Method   | Path                                            | Description                                      |
|----------|-------------------------------------------------|--------------------------------------------------|
| `PUT`    | `/v1/sessions/{id}/chip-grid`                   | Compute and persist chip grid layout             |
| `POST`   | `/v1/sessions/{id}/chip-filters`                | Apply cloud-coverage and variance filters        |
| `GET`    | `/v1/sessions/{id}/chips`                       | Paginated chip list with thumbnail URLs          |
| `GET`    | `/v1/sessions/{id}/chips/{index}/thumbnail.png` | 128x128 PNG thumbnail for a single chip          |
| `GET`    | `/v1/sessions/{id}/manifest.csv`                | Per-chip manifest CSV with bounds and QA stats   |

### Export

| Method   | Path                                            | Description                              |
|----------|-------------------------------------------------|------------------------------------------|
| `POST`   | `/v1/sessions/{id}/export`                      | Start export job, returns `job_id`       |
| `GET`    | `/v1/sessions/{id}/export/{job_id}/download`    | Download completed export ZIP            |

### Jobs

| Method | Path                  | Description                 |
|--------|-----------------------|-----------------------------|
| `GET`  | `/v1/jobs/{job_id}`   | Poll async job status       |
| `GET`  | `/healthz`            | Health check                |

Interactive API documentation is available at http://localhost:8000/docs when the service is running.

## Running Locally

```bash
cd backend
pip install -e .
uvicorn sentinel_backend.api.main:app --reload
```

The server starts on http://localhost:8000.

## Running Tests

```bash
cd backend
pytest
```

## Environment Variables

| Variable           | Default  | Description                                              |
|--------------------|----------|----------------------------------------------------------|
| `DATA_DIR`         | `/data`  | Root directory for session storage                       |
| `SESSION_TTL_HOURS`| `6`      | Hours before an inactive session is reaped               |
