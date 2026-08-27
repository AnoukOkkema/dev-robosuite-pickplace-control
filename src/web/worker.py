"""Run one headless pick-and-place request outside the web-server process."""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import replace
from pathlib import Path

import main as pipeline
from src.util.logging_configurator import LoggingConfigurator
from src.util.system_configurator import SystemConfigurator
from src.web.simulation_jobs import SimulationJobs

jobs = SimulationJobs()


def main(request_path: Path) -> None:
    """Execute a saved API request and write either a result or error file.

    Args:
        request_path: JSON input created by :mod:`src.web.app` for one job.
    """
    request = json.loads(request_path.read_text())
    run_directory = request_path.parent
    try:
        config = SystemConfigurator.load()
        web_config = replace(
            config,
            environment=replace(config.environment, seed=request["seed"]),
            execution=replace(
                config.execution, target_classes=request["target_classes"]
            ),
            visualization=replace(
                config.visualization,
                video_enabled=False,
                trajectory_enabled=True,
                trajectory_path=str(run_directory / "trajectory.bin"),
                trajectory_model_path=str(jobs.model_path()),
            ),
        )
        LoggingConfigurator.setup("web.log")
        result = pipeline.main(
            web_config,
            has_renderer=False,
            pause_flag_path=run_directory / "paused.flag",
        )
        (run_directory / "result.json").write_text(
            json.dumps(
                {
                    "placed_objects": result.placed,
                    "avg_detection_confidence": result.avg_detection_confidence,
                    "avg_pose_position_error_cm": result.avg_pose_position_error_cm,
                    "avg_pose_rotation_error_deg": result.avg_pose_rotation_error_deg,
                    "detection_confidences": result.detection_confidences,
                    "pose_position_errors_cm": result.pose_position_errors_cm,
                    "pose_rotation_errors_deg": result.pose_rotation_errors_deg,
                }
            )
        )
    except Exception as error:
        logging.getLogger(__name__).exception("Website simulation failed")
        (run_directory / "error.json").write_text(json.dumps({"error": str(error)}))
        raise


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python -m src.web.worker REQUEST_JSON")
    main(Path(sys.argv[1]))
