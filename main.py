"""Start the simulated vision-guided robot pick-and-place task.

Run directly for the interactive viewer (``uv run mjpython main.py`` on
macOS, ``uv run python main.py`` on Windows/Linux) or for MP4 recording
(``uv run python -m main`` with ``VIDEO_ENABLED`` on); the README explains
why macOS needs a different interpreter for the two. The web API's worker
(``src/web/worker.py``) imports and calls :func:`main` with its own prepared
config instead.
"""

from src.util.logging_configurator import LoggingConfigurator

# Must run before robosuite is imported (below, transitively via
# PoseDatasetGenerator/PoseVisualizer) -- robosuite emits its startup
# warnings at import time.
LoggingConfigurator.suppress_robosuite_warnings()

from pathlib import Path
from typing import Optional

from src.execution.pickplace_executor import PickPlaceExecutor
from src.recording.factory import build_observers
from src.util.system_configurator import SystemConfigurator
from src.util.types import Config, CropRegion, ImageSize, RunResult
from src.vision.onnx_detector import OnnxDetector
from src.vision.pose_estimator import PoseEstimator


def main(
    config: Optional[Config] = None,
    has_renderer: Optional[bool] = None,
    pause_flag_path: Optional[Path] = None,
) -> RunResult:
    """Run one pick-and-place batch and return its outcome summary.

    Args:
        config: Fully resolved settings, or ``None`` to load config.yaml.
        has_renderer: Whether to open an interactive MuJoCo viewer; ``None``
            opens one exactly when no MP4 is being recorded.
        pause_flag_path: Path of a flag file that pauses the run. After every
            controller step the executor checks this path: as long as a file
            exists there, the simulation waits before stepping again.

    Returns:
        Placed-object count plus this run's average detection confidence and
        average vision-vs-ground-truth position error.
    """
    config = SystemConfigurator.load()
    logger = LoggingConfigurator.setup("main.log")
    logger.info("Pick-and-place pipeline started.")

    detector = OnnxDetector(
        model_path=config.detector.model_path,
        class_names=config.detector.class_names,
        image_size=config.detector.image_size,
    )
    pose_estimator = PoseEstimator(
        model_path=config.pose.model_path,
        num_classes=len(config.detector.class_names),
        pos_image_size=config.pose.pos_image_size,
        rotation_image_size=config.pose.rotation_image_size,
    )

    visualization = config.visualization
    observers = build_observers(visualization, logger=logger)

    if has_renderer is None:
        # Without an explicit choice, open the viewer exactly when no MP4 is
        # being recorded: the interactive viewer and headless video rendering
        # cannot share one process (see README).
        has_renderer = not visualization.video_enabled

    executor = PickPlaceExecutor(
        config=config,
        detector=detector,
        pose_estimator=pose_estimator,
        image_size=ImageSize(height=1080, width=1920),
        crop_region=CropRegion(y1=350, y2=748, x1=400, x2=975),
        has_renderer=has_renderer,
        pause_flag_path=pause_flag_path,
        observers=observers,
    )

    try:
        result = executor.run()
    except Exception:
        logger.exception("Pick-and-place pipeline failed.")
        raise
    finally:
        # Also on failure: release the viewer and MuJoCo resources.
        executor.close()

    logger.info("Pipeline completed | objects_placed=%d", result.placed)
    return result


if __name__ == "__main__":
    main()
