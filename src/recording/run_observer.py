from abc import ABC, abstractmethod

import numpy as np

from src.control.stage import Stage


class BaseRunObserver(ABC):
    """Watches a run from the outside. The executor calls these hooks.

    This keeps recording and visualization out of the pick-and-place code.
    The executor only reports what happens, and observers decide what to
    do with it. The hooks follow the lifecycle of one run:

    ``on_run_start`` -> per scan: ``on_scan``, per object: ``on_target_detected``
    -> ``on_step`` (many) -> ... -> ``on_run_end``
    """

    @abstractmethod
    def on_run_start(self, executor) -> None:
        """Called once per run, after ``env.reset()`` and before the first tick."""

    @abstractmethod
    def on_scan(self, agentview_frame: np.ndarray, poses_cam: list) -> None:
        """Called after every detection scan, whether or not it found anything.

        Args:
            agentview_frame: Full, uncropped BGR agent-view capture the scan
                ran detection on.
            poses_cam: ``(class_name, box, xyz_cam, rot_cam)`` per detected
                target-class object. This is the detection box in cropped
                agent-view pixels plus the camera-frame pose.
        """

    @abstractmethod
    def on_target_detected(self, class_name: str, pose_cam: tuple) -> None:
        """Called when a scan finds the object the run is currently looking for.

        Args:
            class_name: Detected object class.
            pose_cam: ``(class_name, box, xyz_cam, rot_cam)``, the detection
                box in cropped agent-view pixels plus the camera-frame pose.
        """

    @abstractmethod
    def on_step(self, object_name: str, stage: Stage) -> None:
        """Called after every controller step."""

    @abstractmethod
    def on_run_end(self) -> None:
        """Called when the run finishes, also after a failure."""
