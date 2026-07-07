# sentinel-processor — Frontend

Streamlit service that provides the 5-step wizard UI for the sentinel-processor pipeline. The frontend contains no image processing logic; all computation is delegated to the backend REST API via `api_client.py`. Session state is held server-side in the backend; the frontend tracks only the current session ID and wizard step in Streamlit session state.

## Wizard Steps

### Step 1 — Upload

Upload a Sentinel-2 L2A GeoTIFF. Select which bands map to R, G, B and adjust the percentile-stretch clip values. Optionally upload one or more cloudless reference images of the same area for use in radiometric normalisation at the Enhance step. Submitting this step calls `POST /v1/sessions/{id}/source` (and `/references` if provided) then `POST /v1/sessions/{id}/stretch`.

### Step 2 — Dehaze

Enable or skip Dark Channel Prior (DCP) dehazing. Exposes controls for patch size, omega (haze fraction), transmission floor, and guided-filter radius. A brightness histogram shows the before/after pixel distribution. Submitting calls `POST /v1/sessions/{id}/cloud-mask` then `POST /v1/sessions/{id}/dehaze`.

### Step 3 — Enhance

Optional enhancements applied on top of the dehazed image: Gray World white balance, reference-based radiometric normalisation (histogram matching or linear mean/std, requires reference images from Step 1), and CLAHE contrast enhancement. Any combination may be selected, or all skipped. Submitting calls `POST /v1/sessions/{id}/enhance`.

### Step 4 — Chip

Configure the chip grid: chip width and height (pixels), overlap fraction, and edge handling (pad or overlap-shift). Optionally enable cloud-coverage and variance quality filters. A grid preview overlay is rendered on top of the processed image. Submitting calls `PUT /v1/sessions/{id}/chip-grid` and `POST /v1/sessions/{id}/chip-filters`.

### Step 5 — Review & Export

Preview all processing parameters and inspect chips in a paginated tile viewer. Select export options (chip format, naming scheme, z-score normalisation, reference normalisation, COCO/YOLO annotations, whether to include rejected chips) then download the assembled ZIP. Submitting calls `POST /v1/sessions/{id}/export` and polls `GET /v1/jobs/{job_id}` until complete.

## Running Locally

The backend must be running before starting the frontend.

```bash
cd frontend
pip install -r requirements.txt
BACKEND_URL=http://localhost:8000 python -m streamlit run sentinel_frontend/main.py
```

The UI is available at http://localhost:8501.

## Environment Variables

| Variable      | Required | Description                                       |
|---------------|----------|---------------------------------------------------|
| `BACKEND_URL` | Yes      | Base URL of the backend service, e.g. `http://localhost:8000` |

## Running Tests

```bash
cd frontend
pytest
```
