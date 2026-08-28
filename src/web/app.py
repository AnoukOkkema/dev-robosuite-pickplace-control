from __future__ import annotations

import subprocess
import sys

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response

from src.util.types import LiveProgress, SimulationJob, SimulationRequest
from src.web.simulation_jobs import SimulationJobs

app = FastAPI(title="Pick-and-place simulation API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

jobs = SimulationJobs()


@app.get("/api/health")
def health() -> dict[str, bool]:
    """Return a lightweight readiness response without loading MuJoCo."""
    return {"ready": True}


@app.post("/api/simulations", response_model=SimulationJob, status_code=202)
def create_simulation(request: SimulationRequest) -> SimulationJob:
    """Start one isolated headless run and return its identifier immediately."""
    if jobs.is_busy():
        raise HTTPException(
            status_code=409,
            detail="A simulation is already running; wait for it to complete",
        )
    job = jobs.create(
        request.seed, request.target_classes or list(jobs.allowed_classes())
    )
    run_directory = jobs.run_directory(job.id)
    run_directory.mkdir(parents=True, exist_ok=True)
    request_path = run_directory / "request.json"
    request_path.write_text(request.model_dump_json())
    process = subprocess.Popen(
        [sys.executable, "-m", "src.web.worker", str(request_path)],
        cwd=jobs.project_root(),
        start_new_session=True,
    )
    jobs.track_process(job.id, process)
    return job


@app.get("/api/simulations/{job_id}", response_model=SimulationJob)
def simulation_status(job_id: str) -> SimulationJob:
    """Return the latest status for a queued simulation."""
    return jobs.refresh(job_id)


@app.post("/api/simulations/{job_id}/pause", response_model=SimulationJob)
def pause_simulation(job_id: str) -> SimulationJob:
    """Block the run between controller ticks until resumed or stopped."""
    job = jobs.refresh(job_id)
    if job.status not in ("queued", "running"):
        raise HTTPException(
            status_code=409, detail=f"Cannot pause a {job.status} simulation"
        )
    jobs.pause_flag_path(job.id).touch()
    return jobs.refresh(job_id)


@app.post("/api/simulations/{job_id}/resume", response_model=SimulationJob)
def resume_simulation(job_id: str) -> SimulationJob:
    """Let a paused run continue from where it was paused."""
    job = jobs.refresh(job_id)
    jobs.pause_flag_path(job.id).unlink(missing_ok=True)
    return jobs.refresh(job_id)


@app.post("/api/simulations/{job_id}/stop", response_model=SimulationJob)
def stop_simulation(job_id: str) -> SimulationJob:
    """Terminate a run in progress. Whatever it streamed live is kept.

    Unlike a crash, this is a requested stop. The job is reported as
    "stopped" rather than "failed". The player is expected to fall back
    to its partial live view instead of fetching a final trajectory that
    was never written.
    """
    job = jobs.get(job_id)
    if job.status in ("completed", "failed", "stopped"):
        raise HTTPException(status_code=409, detail=f"Simulation already {job.status}")
    jobs.stop(job_id)
    return jobs.refresh(job_id)


@app.get("/api/simulations/{job_id}/trajectory")
def simulation_trajectory(job_id: str) -> FileResponse:
    """Return the compact trajectory once the requested run is complete."""
    job = jobs.refresh(job_id)
    if job.status != "completed":
        raise HTTPException(status_code=409, detail="Simulation is not complete")

    trajectory_path = jobs.run_directory(job.id) / "trajectory.bin"
    if not trajectory_path.is_file():
        raise HTTPException(status_code=500, detail="Simulation trajectory is missing")
    return FileResponse(
        trajectory_path,
        media_type="application/octet-stream",
        filename=f"{job.id}.trajectory.bin",
    )


@app.get("/api/simulations/{job_id}/live", response_model=LiveProgress)
def simulation_live_progress(job_id: str) -> LiveProgress:
    """Return how many trajectory frames a running simulation has streamed.

    Unlike ``/trajectory``, this works before the job completes. The
    player polls it to grow a live view instead of waiting for the run to
    finish.
    """
    job = jobs.refresh(job_id)
    meta = jobs.read_live_meta(job_id)
    if meta is None:
        return LiveProgress(ready=False, status=job.status)

    live_path = jobs.run_directory(job_id) / "live.bin"
    frame_count = (
        live_path.stat().st_size // jobs.live_frame_stride(meta["nq"])
        if live_path.is_file()
        else 0
    )
    return LiveProgress(
        ready=True,
        status=job.status,
        nq=meta["nq"],
        frame_count=frame_count,
        object_names=meta["object_names"],
        stage_names=meta["stage_names"],
    )


@app.get("/api/simulations/{job_id}/live/frames")
def simulation_live_frames(
    job_id: str, from_: int = Query(default=0, alias="from")
) -> Response:
    """Return raw, newly-available live frames starting at frame ``from_``.

    The response is a byte-exact slice of ``live.bin``. See
    :meth:`TrajectoryRecorder._append_live_frame` for the record layout.
    It's truncated to whole frames so a partially-written trailing record
    is never served.
    """
    jobs.get(job_id)
    meta = jobs.read_live_meta(job_id)
    if meta is None:
        return Response(content=b"", media_type="application/octet-stream")

    live_path = jobs.run_directory(job_id) / "live.bin"
    if not live_path.is_file() or from_ < 0:
        return Response(content=b"", media_type="application/octet-stream")

    stride = jobs.live_frame_stride(meta["nq"])
    frame_count = live_path.stat().st_size // stride
    start_frame = min(from_, frame_count)
    with live_path.open("rb") as live_file:
        live_file.seek(start_frame * stride)
        chunk = live_file.read((frame_count - start_frame) * stride)
    return Response(content=chunk, media_type="application/octet-stream")


@app.get("/api/model")
def model() -> FileResponse:
    """Return the cached model package shared by every browser run."""
    path = jobs.model_path()
    if not path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Model is created by the first completed simulation",
        )
    return FileResponse(
        path,
        media_type="application/zip",
        filename="scene-v1.zip",
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )
