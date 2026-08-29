import logging

from src.control.stage import Stage
from src.recording.run_observer import BaseRunObserver
from src.recording.trajectory_recorder import TrajectoryRecorder
from src.util.types import RunResult


class TrajectoryObserver(BaseRunObserver):
    """Record a run for the browser player by listening to the executor.

    The executor announces three moments; each one triggers the matching
    :class:`TrajectoryRecorder` call:

    * run start -> package the 3D scene (``model.zip``)
    * every controller step -> buffer the current simulation state
    * run end -> write the complete recording (``trajectory.bin``) and its
      result summary (``<stem>.result.json``)
    """

    def __init__(self, trajectory_path: str, model_path: str, logger=None) -> None:
        """Create the observer and its underlying recorder.

        Args:
            trajectory_path: Destination of the final ``trajectory.bin``.
            model_path: Destination of the packaged scene ``model.zip``.
            logger: Logger used to report the exported files.
        """
        self._recorder = TrajectoryRecorder(
            trajectory_path,
            model_path,
            logger=logger or logging.getLogger(__name__),
        )
        self._executor = None

    def on_run_start(self, executor) -> None:
        """Package the scene model.

        This runs before the first controller tick, so a browser player can
        download and render the scene while the run is still going, rather
        than waiting for it to finish.

        Args:
            executor: The executor running the pick-and-place episode, used
                to reach the compiled MuJoCo model for export.
        """
        self._executor = executor
        self._recorder.export_model(executor.env.sim.model)

    def on_scan(self, agentview_frame, poses_cam: list) -> None:
        """Trajectory recording has no use for scan images.

        Args:
            agentview_frame: Full, uncropped BGR agent-view capture the scan
                ran detection on.
            poses_cam: ``(class_name, box, xyz_cam, rot_cam)`` per detected
                target-class object.
        """

    def on_target_detected(self, class_name: str, pose_cam: tuple) -> None:
        """Trajectory recording has no use for the detected pose.

        Args:
            class_name: Detected object class.
            pose_cam: ``(class_name, box, xyz_cam, rot_cam)`` detection box
                in cropped agent-view pixels plus the camera-frame pose.
        """

    def on_step(self, object_name: str, stage: Stage) -> None:
        """Capture the simulation state the controller step just produced.

        Args:
            object_name: Object the controller is currently handling,
                recorded alongside the simulation state.
            stage: Current pick-and-place stage; recorded by name alongside
                the simulation state.
        """
        data = self._executor.env.sim.data
        self._recorder.capture(
            data.time,
            data.qpos,
            object_name=object_name,
            stage_name=stage.name,
        )

    def on_run_end(self, result: RunResult) -> None:
        """Write the final trajectory and its result summary.

        The executor calls this in a ``finally``, so even a crashed run
        still exports every state captured up to that point, alongside a
        summary of whatever was collected before the crash.

        Args:
            result: Placement/vision-accuracy summary written out alongside
                the trajectory.
        """
        self._recorder.export(self._executor.env.sim.model)
        self._recorder.write_result(result)
