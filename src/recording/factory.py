from src.recording.run_observer import BaseRunObserver
from src.recording.scan_image_observer import ScanImageObserver
from src.recording.trajectory_observer import TrajectoryObserver
from src.recording.video_observer import VideoObserver
from src.util.types import VisualizationConfig


def build_observers(
    visualization: VisualizationConfig, logger=None
) -> list[BaseRunObserver]:
    """Construct the observers ``visualization`` enables, in no particular order.

    Args:
        visualization: Config selecting which observers to build and their
            output paths and settings.
        logger: Logger passed to every constructed observer.
    """
    observers: list[BaseRunObserver] = [
        TrajectoryObserver(
            visualization.trajectory_path,
            visualization.trajectory_model_path,
            logger=logger,
        )
    ]
    if visualization.video_enabled:
        observers.append(
            VideoObserver(
                visualization.video_path,
                visualization.video_fps,
                visualization.video_capture_every_ticks,
                logger=logger,
            )
        )
    if visualization.scan_images_enabled:
        observers.append(
            ScanImageObserver(visualization.scan_images_dir, logger=logger)
        )
    return observers
