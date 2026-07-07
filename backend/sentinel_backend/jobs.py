"""In-process async job registry: create, track, and run background processing tasks."""
import asyncio, uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Any
from sentinel_backend.models import JobRecord, JobStatus

_jobs: dict[str, JobRecord] = {}
_executor = ThreadPoolExecutor(max_workers=2)


def create_job() -> str:
    job_id = str(uuid.uuid4())
    _jobs[job_id] = JobRecord(job_id=job_id, status=JobStatus.queued)
    return job_id


def get_job(job_id: str) -> JobRecord | None:
    return _jobs.get(job_id)


async def run_job(job_id: str, fn: Callable, *args, **kwargs) -> None:
    """Run fn(*args, **kwargs) in a thread; update job record on completion."""
    _jobs[job_id] = JobRecord(job_id=job_id, status=JobStatus.running)
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(_executor, lambda: fn(*args, **kwargs))
        _jobs[job_id] = JobRecord(job_id=job_id, status=JobStatus.done,
                                   progress=1.0, result=result)
    except Exception as e:
        _jobs[job_id] = JobRecord(job_id=job_id, status=JobStatus.error,
                                   error=str(e))
