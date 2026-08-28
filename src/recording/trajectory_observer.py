import logging

from src.control.stage import Stage
from src.recording.run_observer import BaseRunObserver
from src.recording.trajectory_recorder import TrajectoryRecorder


class TrajectoryObserver(BaseRunObserver):
    """Record a run for the browser player by listening to the executor.

    The executor announces three moments; each one triggers the matching
    :class:`TrajectoryRecorder` call:

    * run start -> package the 3D scene (``model.zip``) and open the live
      stream
    * every controller step -> append the current simulation state
      (``live.bin``)
    * run end -> write the complete recording (``trajectory.bin``)
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
        """Package the scene model and open the live stream.

        This runs before the first controller tick, so a browser player can
        show the scene and start following frames within seconds instead of
        only after the full run.
        """
        self._executor = executor
        self._recorder.export_model(executor.env.sim.model)
        self._recorder.start(executor.env.sim.model.nq, executor.target_classes)

    def on_scan(self, agentview_frame, poses_cam: list) -> None:
        """Trajectory recording has no use for scan images."""

    def on_target_detected(self, class_name: str, pose_cam: tuple) -> None:
        """Trajectory recording has no use for the detected pose."""

    def on_step(self, object_name: str, stage: Stage) -> None:
        """Capture the simulation state the controller step just produced."""
        data = self._executor.env.sim.data
        self._recorder.capture(
            data.time,
            data.qpos,
            object_name=object_name,
            stage_name=stage.name,
        )

    def on_run_end(self) -> None:
        """Write the final trajectory.

        The executor calls this in a ``finally``, so even a crashed run
        still exports every state captured up to that point.
        """
        self._recorder.export(self._executor.env.sim.model)
