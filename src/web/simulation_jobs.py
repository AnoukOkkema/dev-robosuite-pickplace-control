"""Job registry for headless pick-and-place simulations queued through the web API."""

from __future__ import annotations

import json
import shutil
import subprocess
import uuid
from datetime import datetime, timedelta
from pathlib import Path

from fastapi import HTTPException

from src.util.system_configurator import SystemConfigurator
from src.util.types import SimulationJob


class SimulationJobs:
    """Tracks queued, running, and finished simulation jobs and their worker process."""

    _MAX_RUN_AGE = timedelta(days=1)

    def __init__(self) -> None:
        self._output_root = self.project_root() / "outputs" / "web"
        self._jobs: dict[str, SimulationJob] = {}
        self._processes: dict[str, subprocess.Popen[bytes]] = {}
        self._stopped_job_ids: set[str] = set()

    @staticmethod
    def project_root() -> Path:
        """Return the repository root, resolved from this file's own location."""
        return Path(__file__).resolve().parents[2]

    def model_path(self) -> Path:
        """Return the cached, reusable scene model path shared by every run."""
        return self._output_root / "models" / "scene-v1.zip"

    @staticmethod
    def allowed_classes() -> tuple[str, ...]:
        """Return the object classes the configured detector can recognize."""
        return tuple(SystemConfigurator.load().detector.class_names)

    def is_busy(self) -> bool:
        """Return whether a worker process is currently running."""
        return any(process.poll() is None for process in self._processes.values())

    def create(self, seed: int | None, target_classes: list[str]) -> SimulationJob:
        """Register a new queued job and return it."""
        self._cleanup_old_runs()
        job = SimulationJob(
            id=uuid.uuid4().hex,
            status="queued",
            seed=seed,
            target_classes=target_classes,
        )
        self._jobs[job.id] = job
        return job

    def _cleanup_old_runs(self) -> None:
        """Delete run directories whose contents haven't changed in a day."""
        runs_root = self._output_root / "runs"
        if not runs_root.is_dir():
            return
        cutoff = datetime.now().timestamp() - self._MAX_RUN_AGE.total_seconds()
        for run_directory in runs_root.iterdir():
            if run_directory.is_dir() and run_directory.stat().st_mtime < cutoff:
                shutil.rmtree(run_directory, ignore_errors=True)

    def track_process(self, job_id: str, process: subprocess.Popen[bytes]) -> None:
        """Associate a spawned worker process with its job."""
        self._processes[job_id] = process

    def get(self, job_id: str) -> SimulationJob:
        """Return a job or convert an unknown identifier to a 404 response."""
        try:
            return self._jobs[job_id]
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Unknown simulation") from error

    def refresh(self, job_id: str) -> SimulationJob:
        """Merge a completed worker's result into the in-memory job record."""
        job = self.get(job_id)
        run_directory = self.run_directory(job.id)
        result_path = run_directory / "result.json"
        error_path = run_directory / "error.json"
        if result_path.is_file():
            result = json.loads(result_path.read_text())
            job.status = "completed"
            job.placed_objects = result["placed_objects"]
            job.avg_detection_confidence = result.get("avg_detection_confidence")
            job.avg_pose_position_error_cm = result.get("avg_pose_position_error_cm")
            job.detection_confidences = result.get("detection_confidences")
            job.pose_position_errors_cm = result.get("pose_position_errors_cm")
        elif error_path.is_file():
            result = json.loads(error_path.read_text())
            job.status = "failed"
            job.error = result["error"]
        elif self._processes[job.id].poll() is None:
            job.status = (
                "paused" if self.pause_flag_path(job.id).is_file() else "running"
            )
        elif job.id in self._stopped_job_ids:
            job.status = "stopped"
        else:
            job.status = "failed"
            job.error = "Simulation worker exited before writing a result"
        return job

    def stop(self, job_id: str) -> None:
        """Terminate a job's worker process. Whatever it streamed live is kept."""
        self._stopped_job_ids.add(job_id)
        self._processes[job_id].terminate()
        # don't leave a stale flag file behind
        self.pause_flag_path(job_id).unlink(missing_ok=True)

    def run_directory(self, job_id: str) -> Path:
        """Return the per-job output directory used by both the API and worker."""
        return self._output_root / "runs" / job_id

    def pause_flag_path(self, job_id: str) -> Path:
        """Return the flag file ``PickPlaceExecutor._wait_while_paused`` polls."""
        return self.run_directory(job_id) / "paused.flag"

    def read_live_meta(self, job_id: str) -> dict | None:
        """Return the live-stream header, or ``None`` before the worker writes it.

        See ``TrajectoryRecorder.start`` -- the header appears within seconds of
        a run starting, well before its first frame.
        """
        live_meta_path = self.run_directory(job_id) / "live.meta.json"
        if not live_meta_path.is_file():
            return None
        return json.loads(live_meta_path.read_text())

    @staticmethod
    def live_frame_stride(nq: int) -> int:
        """Return the byte size of one live-stream record for a given ``nq``.

        Matches ``TrajectoryRecorder._append_live_frame``: ``time:f32,
        qpos:f32[nq], object:u8, stage:u8, pad:u8[2]``.
        """
        return (nq + 2) * 4
