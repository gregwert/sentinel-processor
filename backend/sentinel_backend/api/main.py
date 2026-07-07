"""FastAPI application factory: routes, lifespan session reaper, and job status endpoint."""
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from sentinel_backend.api.routers import sessions, processing, chips, export
from sentinel_backend.storage import reap_expired_sessions
from sentinel_backend.jobs import get_job
from sentinel_backend.models import JobRecord

_reaper_task = None


@asynccontextmanager
async def lifespan(app_: FastAPI):
    global _reaper_task

    async def reaper():
        while True:
            await asyncio.sleep(3600)
            reap_expired_sessions()

    _reaper_task = asyncio.create_task(reaper())
    yield
    if _reaper_task:
        _reaper_task.cancel()


app = FastAPI(title="sentinel-processor backend", version="1.0.0", lifespan=lifespan)

app.include_router(sessions.router,   prefix="/v1/sessions", tags=["sessions"])
app.include_router(processing.router, prefix="/v1/sessions", tags=["processing"])
app.include_router(chips.router,      prefix="/v1/sessions", tags=["chips"])
app.include_router(export.router,     prefix="/v1/sessions", tags=["export"])


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.get("/v1/jobs/{job_id}", response_model=JobRecord, tags=["jobs"])
async def get_job_status(job_id: str) -> JobRecord:
    record = get_job(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return record
