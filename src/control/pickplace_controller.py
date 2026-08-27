"""Run the ordered robot stages needed to pick one object and place it in a bin.

This module is the task-level controller. It chooses the next Cartesian goal
for each named stage, while :mod:`joint_motion` converts that goal into small
absolute Panda joint-position actions.
"""

import logging
from collections.abc import Callable

import numpy as np

from src.control.joint_motion import JointMotion
from src.control.stage import Stage
from src.geometry.coordinate_transform import CoordinateTransformer
from src.util.types import (
    ControllerRunState,
    MotionConfig,
    PlacementTarget,
    PoseCommand,
    StagesConfig,
    WristTargets,
)


class PickPlaceController:
    """Move one detected object from the table to its assigned bin.

    Stages execute in this order: raise above the table, align over the object,
    descend, orient the gripper when required, grasp, lift, re-zero yaw, move
    over the bin, descend to its release height, and open the gripper. The
    controller advances only after a stage's position, wrist, and yaw targets
    have converged.

    Attributes:
        env: Robosuite environment that receives joint-position actions.
        motion_config: Controller gains and per-tick motion limits.
        stages_config: Stage targets, tolerances, and dwell durations.
        logger: Logger used for stage progress and diagnostics.
        motion: Joint-motion helper used to build actions and wrist targets.
    """

    def __init__(
        self,
        env,
        motion_config: MotionConfig,
        stages_config: StagesConfig,
        logger=None,
        frame_callback: Callable[[str, Stage], None] | None = None,
    ) -> None:
        """Initialize the controller.

        Args:
            env: Robosuite environment that receives joint-position actions.
            motion_config: Controller gains and motion limits.
            stages_config: Stage targets, tolerances, and dwell durations.
            logger: Logger used for stage progress and diagnostics.
            frame_callback: Optional callback invoked after each simulation step.
        """
        self.env = env
        self.motion_config = motion_config
        self.stages_config = stages_config
        self.logger = logger or logging.getLogger(__name__)
        self.motion = JointMotion(env, motion_config)
        self._frame_callback = frame_callback
        self._active_class_name: str | None = None
        self._active_stage: Stage | None = None

    @property
    def open_gripper(self) -> float:
        """Return the action value that opens the gripper."""
        return self.stages_config.open_gripper

    @property
    def closed_gripper(self) -> float:
        """Return the action value that closes the gripper."""
        return self.stages_config.closed_gripper

    @staticmethod
    def _closest_symmetric_yaw(target_yaw: float, current_yaw: float) -> float:
        """Return the equivalent target yaw requiring the smallest turn."""
        return min(
            (target_yaw - 180.0, target_yaw, target_yaw + 180.0),
            key=lambda value: abs(value - current_yaw),
        )

    def run(
        self,
        obs: dict,
        target: PlacementTarget,
        is_last: bool = False,
        reset_osc: bool = False,
    ) -> tuple[dict, bool]:
        """Execute the complete pick-and-place sequence for one detected object.

        Args:
            obs: Most recent environment observation.
            target: Detected object pose and its destination-bin pose.
            is_last: Retained for compatibility with executor implementations.
            reset_osc: Reset the Robosuite composite controller before the run.

        Returns:
            The final simulation observation and ``True`` after release.

        Raises:
            RuntimeError: If a stage exceeds its tick limit or is unknown.
        """

        del is_last
        if reset_osc:
            # Robosuite still owns the composite controller under our joint actions.
            self.env.robots[0].composite_controller.reset()
        self.motion.reset_posture()

        start_position, home_orientation = self.motion.eef_pose()
        state = ControllerRunState(stage=Stage.RAISE)
        self._active_class_name = target.class_name
        self._active_stage = state.stage
        release_heights = self.stages_config.bin_release_heights
        release_height = release_heights[target.class_name]
        object_start_z = self._object_position(target)[2]
        announced_stage = None

        self.logger.info(
            "%s | joint-pose start | object=(%.3f, %.3f, %.3f) | "
            "bin=(%.3f, %.3f, %.3f)",
            target.class_name,
            *target.object_pos,
            *target.bin_pos,
        )

        while True:
            position, orientation = self.motion.eef_pose()

            if state.stage == Stage.RAISE:
                command = self._pose_command(
                    [
                        start_position[0],
                        start_position[1],
                        self.stages_config.approach_height,
                    ],
                    home_orientation,
                    self.open_gripper,
                    keep_world_yaw=True,
                )
                wrist_targets = self.motion.vertical_down_wrist_targets()
                next_stage = Stage.ALIGN_XY

            elif state.stage == Stage.ALIGN_XY:
                command = self._pose_command(
                    [
                        target.object_pos[0],
                        target.object_pos[1],
                        self.stages_config.approach_height,
                    ],
                    home_orientation,
                    self.open_gripper,
                    keep_world_yaw=True,
                )
                wrist_targets = self.motion.vertical_down_wrist_targets()
                next_stage = Stage.DESCEND

            elif state.stage == Stage.DESCEND:
                command = self._pose_command(
                    target.object_pos
                    + [0.0, 0.0, self.stages_config.descend_clearance],
                    home_orientation,
                    self.open_gripper,
                    keep_world_yaw=True,
                )
                wrist_targets = self.motion.vertical_down_wrist_targets()
                # The cylindrical can is grasped without object-specific yaw alignment.
                next_stage = (
                    Stage.FINE_DESCEND
                    if target.class_name == "Can"
                    else Stage.ORIENT_YAW
                )

            elif state.stage == Stage.ORIENT_YAW:
                current_yaw = self._yaw_deg(orientation)
                desired_yaw = self._closest_symmetric_yaw(
                    target.yaw_deg + self.stages_config.grasp_yaw_offset_deg,
                    current_yaw,
                )
                if state.stage != announced_stage:
                    self._log_yaw_stage_started(
                        target.class_name, state.stage, desired_yaw
                    )
                    announced_stage = state.stage
                obs = self._run_yaw_stage(
                    obs,
                    state,
                    orientation,
                    desired_yaw,
                    self.open_gripper,
                    Stage.FINE_DESCEND,
                    target.class_name,
                )
                continue

            elif state.stage == Stage.FINE_DESCEND:
                fine_offset = self.stages_config.fine_descend_offsets.get(
                    target.class_name, 0.0
                )
                fine_max_delta = (
                    self.stages_config.fine_descend_max_position_deltas.get(
                        target.class_name
                    )
                )
                command = self._pose_command(
                    target.object_pos + [0.0, 0.0, fine_offset],
                    home_orientation,
                    self.open_gripper,
                    joint7_target=state.grasp_joint7_target,
                    max_position_delta=fine_max_delta,
                )
                # Preserve the grasp orientation while moving to the contact height.
                wrist_targets = self.motion.vertical_down_wrist_targets()
                next_stage = Stage.GRASP

            elif state.stage == Stage.GRASP:
                if state.stage != announced_stage:
                    self.logger.info(
                        "%s | GRASP started | closing gripper | dwell_ticks=%d",
                        target.class_name,
                        self.stages_config.grasp_dwell_ticks,
                    )
                    announced_stage = state.stage
                obs = self._hold_gripper(
                    self.stages_config.grasp_dwell_ticks, self.closed_gripper
                )
                if not self._is_object_grasped(target):
                    self.logger.error(
                        "%s | GRASP failed | object is not held by the gripper",
                        target.class_name,
                    )
                    return obs, False
                self.logger.info(
                    "%s | GRASP completed | object is held | dwell_ticks=%d",
                    target.class_name,
                    self.stages_config.grasp_dwell_ticks,
                )
                self._enter_stage(state, Stage.LIFT)
                continue

            elif state.stage == Stage.LIFT:
                command = self._pose_command(
                    [
                        target.object_pos[0],
                        target.object_pos[1],
                        self.stages_config.approach_height,
                    ],
                    home_orientation,
                    self.closed_gripper,
                    joint7_target=state.grasp_joint7_target,
                )
                wrist_targets = self.motion.vertical_down_wrist_targets()
                next_stage = Stage.REZERO_YAW

            elif state.stage == Stage.REZERO_YAW:
                current_yaw = self._yaw_deg(orientation)
                desired_yaw = (
                    current_yaw
                    if target.class_name == "Can"
                    else self._closest_symmetric_yaw(90.0, current_yaw)
                )
                if state.stage != announced_stage:
                    self._log_yaw_stage_started(
                        target.class_name, state.stage, desired_yaw
                    )
                    announced_stage = state.stage
                obs = self._run_yaw_stage(
                    obs,
                    state,
                    orientation,
                    desired_yaw,
                    self.closed_gripper,
                    Stage.MOVE_TO_BIN,
                    target.class_name,
                    record_rezero_pose=True,
                )
                continue

            elif state.stage == Stage.MOVE_TO_BIN:
                command = self._pose_command(
                    [
                        target.bin_pos[0],
                        target.bin_pos[1],
                        self.stages_config.approach_height,
                    ],
                    self._required_rezero_orientation(state),
                    self.closed_gripper,
                    keep_world_yaw=True,
                    yaw_target_deg=self._required_rezero_yaw(state),
                )
                # Keep the re-zeroed world yaw rather than a fixed Joint-7 angle.
                wrist_targets = self.motion.vertical_down_wrist_targets()
                next_stage = Stage.DESCEND_TO_BIN

            elif state.stage == Stage.DESCEND_TO_BIN:
                command = self._pose_command(
                    [target.bin_pos[0], target.bin_pos[1], release_height],
                    self._required_rezero_orientation(state),
                    self.closed_gripper,
                    position_tolerance=self.stages_config.bin_release_tolerance,
                    keep_world_yaw=True,
                    yaw_target_deg=self._required_rezero_yaw(state),
                )
                wrist_targets = self.motion.vertical_down_wrist_targets()
                next_stage = Stage.RELEASE

            elif state.stage == Stage.RELEASE:
                if state.stage != announced_stage:
                    self.logger.info(
                        "%s | RELEASE started | opening gripper | dwell_ticks=%d",
                        target.class_name,
                        self.stages_config.release_dwell_ticks,
                    )
                    announced_stage = state.stage
                obs = self._hold_gripper(
                    self.stages_config.release_dwell_ticks, self.open_gripper
                )
                self.logger.info(
                    "%s | RELEASE completed | dwell_ticks=%d | placement finished",
                    target.class_name,
                    self.stages_config.release_dwell_ticks,
                )
                return obs, True

            else:
                raise RuntimeError(f"Unexpected stage: {state.stage.name}")

            if state.stage != announced_stage:
                self._log_pose_stage_started(target.class_name, state.stage, command)
                announced_stage = state.stage

            if self._pose_converged(command, position, wrist_targets, orientation):
                if state.stage == Stage.LIFT:
                    if not self._object_lifted(target, object_start_z):
                        object_z = self._object_position(target)[2]
                        required_z = (
                            object_start_z + self.stages_config.grasp_min_lift_height
                        )
                        self.logger.error(
                            "%s | LIFT failed | object was not lifted | "
                            "start_z=%.3f m | actual_z=%.3f m | required_z=%.3f m",
                            target.class_name,
                            object_start_z,
                            object_z,
                            required_z,
                        )
                        return obs, False
                    self.logger.info(
                        "%s | LIFT verified | object rose %.3f m while grasped",
                        target.class_name,
                        self._object_position(target)[2] - object_start_z,
                    )
                self._log_stage_completed(
                    target.class_name,
                    state.stage,
                    state.stage_tick,
                    command,
                    position,
                    wrist_targets,
                    orientation,
                )
                self._enter_stage(state, next_stage)
                continue

            obs = self._step_pose(command, wrist_targets)
            self._increment_stage_tick(state, target.class_name, command, position)

    def _log_stage_completed(
        self,
        class_name: str,
        stage: Stage,
        ticks: int,
        command: PoseCommand,
        position: np.ndarray,
        wrist_targets: WristTargets,
        orientation: np.ndarray,
    ) -> None:
        """Log a concise, human-readable result for one movement stage."""
        position_error = np.linalg.norm(command.position - position)
        joints = self.motion.joint_positions
        fields = [f"position_error={position_error:.4f} m"]
        if wrist_targets.joint5 is not None:
            joint5_error = np.degrees(
                abs(joints[self.motion.joint5_index] - wrist_targets.joint5)
            )
            fields.append(f"joint5_error={joint5_error:.2f}°")
        joint6_error = np.degrees(
            abs(joints[self.motion.joint6_index] - wrist_targets.joint6)
        )
        fields.append(f"joint6_error={joint6_error:.2f}°")
        if command.yaw_target_deg is not None:
            yaw_error = self._yaw_error(command.yaw_target_deg, orientation)
            fields.append(f"yaw_error={yaw_error:.2f}°")
        self.logger.info(
            "%s | %s completed | ticks=%d | %s",
            class_name,
            stage.name,
            ticks,
            " | ".join(fields),
        )

    def _log_pose_stage_started(
        self,
        class_name: str,
        stage: Stage,
        command: PoseCommand,
    ) -> None:
        """Log the world-frame target that a Cartesian stage will approach."""
        self.logger.info(
            "%s | %s started | target_position=(%.3f, %.3f, %.3f)",
            class_name,
            stage.name,
            *command.position,
        )

    def _log_yaw_stage_started(
        self,
        class_name: str,
        stage: Stage,
        target_yaw: float,
    ) -> None:
        """Log the desired world yaw for a yaw-only stage."""
        self.logger.info(
            "%s | %s started | target_yaw=%.2f°",
            class_name,
            stage.name,
            target_yaw,
        )

    def _pose_command(
        self,
        position: np.ndarray | list[float],
        orientation: np.ndarray,
        gripper_action: float,
        position_tolerance: float | None = None,
        joint7_target: float | None = None,
        max_position_delta: float | None = None,
        keep_world_yaw: bool = False,
        yaw_target_deg: float | None = None,
    ) -> PoseCommand:
        """Create the pose goal used by a Cartesian movement stage.

        A command combines a world-frame position, desired gripper orientation,
        gripper state, and optional per-object limits such as a slower final
        descend. ``keep_world_yaw`` enables only Z-axis orientation correction.
        """
        return PoseCommand(
            position=np.asarray(position, dtype=float),
            orientation=orientation,
            gripper_action=gripper_action,
            position_tolerance=position_tolerance
            or self.stages_config.position_tolerance,
            max_position_delta=max_position_delta,
            joint7_target=joint7_target,
            orientation_weight=1.0 if keep_world_yaw else None,
            rotation_axis_weights=np.array([0.0, 0.0, 1.0]) if keep_world_yaw else None,
            yaw_target_deg=yaw_target_deg,
        )

    def _run_yaw_stage(
        self,
        obs: dict,
        state: ControllerRunState,
        orientation: np.ndarray,
        desired_yaw: float,
        gripper_action: float,
        next_stage: Stage,
        class_name: str,
        record_rezero_pose: bool = False,
    ) -> dict:
        """Rotate only Joint 7 toward a desired world yaw.

        When the yaw error is within tolerance, record the resulting grasp or
        re-zero pose and enter the next stage. Otherwise, apply one bounded
        yaw action and remain in the current stage.
        """
        current_yaw = self._yaw_deg(orientation)
        yaw_error = self._yaw_error(desired_yaw, orientation)
        if abs(yaw_error) < self.stages_config.yaw_tolerance_deg:
            if record_rezero_pose:
                state.rezero_yaw_deg = current_yaw
                state.rezero_orientation = orientation.copy()
            else:
                state.grasp_joint7_target = self.motion.joint_positions[
                    self.motion_config.yaw_joint_index
                ]
            self.logger.info(
                "%s | %s completed | ticks=%d | yaw_error=%.2f°",
                class_name,
                state.stage.name,
                state.stage_tick,
                yaw_error,
            )
            self._enter_stage(state, next_stage)
            return obs

        obs = self._step(self.motion.joint7_yaw_action(yaw_error, gripper_action))
        self._increment_stage_tick(state, class_name)
        return obs

    def _pose_converged(
        self,
        command: PoseCommand,
        position: np.ndarray,
        wrist_targets: WristTargets,
        orientation: np.ndarray,
    ) -> bool:
        """Return whether the pose command and optional yaw target converged."""
        position_ok = (
            np.linalg.norm(command.position - position) < command.position_tolerance
        )
        wrist_ok = self._wrist_converged(wrist_targets)
        if command.yaw_target_deg is None:
            return position_ok and wrist_ok
        yaw_error = self._yaw_error(command.yaw_target_deg, orientation)
        return (
            position_ok
            and wrist_ok
            and abs(yaw_error) < self.stages_config.yaw_tolerance_deg
        )

    def _wrist_converged(self, targets: WristTargets) -> bool:
        """Return whether the commanded vertical-down wrist targets converged."""
        joints = self.motion.joint_positions
        joint6_error = abs(joints[self.motion.joint6_index] - targets.joint6)
        joint6_ok = joint6_error < np.deg2rad(
            self.motion_config.joint6_down_tolerance_deg
        )
        if targets.joint5 is None:
            return joint6_ok
        joint5_error = abs(joints[self.motion.joint5_index] - targets.joint5)
        return joint6_ok and joint5_error < np.deg2rad(
            self.motion_config.joint5_down_tolerance_deg
        )

    def _step_pose(self, command: PoseCommand, wrist_targets: WristTargets) -> dict:
        """Build and apply one joint-position action for a Cartesian command."""
        action = self.motion.action(
            command.position,
            command.orientation,
            command.gripper_action,
            max_position_delta=command.max_position_delta,
            joint5_target=wrist_targets.joint5,
            joint6_target=wrist_targets.joint6,
            joint7_target=command.joint7_target,
            orientation_weight=command.orientation_weight,
            rotation_axis_weights=command.rotation_axis_weights,
        )
        return self._step(action)

    def _hold_gripper(self, ticks: int, gripper_action: float) -> dict:
        """Keep the current arm pose while applying a gripper action for ``ticks``."""
        obs = None
        for _ in range(ticks):
            obs = self._step(
                np.concatenate((self.motion.joint_positions, [gripper_action]))
            )
        return obs

    def _step(self, action: np.ndarray) -> dict:
        """Apply one action and keep viewer geometry settings consistent."""
        obs, _, _, _ = self.env.step(action)
        native_viewer = getattr(getattr(self.env, "viewer", None), "viewer", None)
        if native_viewer is not None:
            native_viewer.opt.geomgroup[0] = 0
            native_viewer.opt.geomgroup[1] = 1
        if self._frame_callback is not None and self._active_stage is not None:
            self._frame_callback(self._active_class_name, self._active_stage)
        return obs

    def _enter_stage(self, state: ControllerRunState, next_stage: Stage) -> None:
        """Set the next stage and reset its elapsed tick counter."""
        state.stage = next_stage
        state.stage_tick = 0
        self._active_stage = next_stage

    def _increment_stage_tick(
        self,
        state: ControllerRunState,
        class_name: str,
        command: PoseCommand | None = None,
        position: np.ndarray | None = None,
    ) -> None:
        """Increment the active-stage counter and raise on non-convergence."""
        state.stage_tick += 1
        self._log_stage_progress(state, class_name, command, position)
        if state.stage_tick < self.stages_config.max_ticks_per_stage:
            return
        # Stop instead of issuing unbounded corrections when a target is unreachable.
        message = f"{class_name} | {state.stage.name} | STAGE not converged"
        if command is not None and position is not None:
            message += (
                f" | position_error={np.linalg.norm(command.position - position):.4f} m"
            )
        raise RuntimeError(message)

    def _log_stage_progress(
        self,
        state: ControllerRunState,
        class_name: str,
        command: PoseCommand | None,
        position: np.ndarray | None,
    ) -> None:
        """Log periodic DEBUG progress without cluttering normal run output."""
        if state.stage_tick % self.stages_config.log_every_ticks:
            return
        if command is None or position is None:
            self.logger.debug(
                "%s | %s progress | tick=%d",
                class_name,
                state.stage.name,
                state.stage_tick,
            )
            return
        self.logger.debug(
            "%s | %s progress | tick=%d | position_error=%.4f m",
            class_name,
            state.stage.name,
            state.stage_tick,
            np.linalg.norm(command.position - position),
        )

    def _object_position(self, target: PlacementTarget) -> np.ndarray:
        """Return the simulated world position of the object being moved."""
        body_id = self.env.obj_body_id[target.class_name]
        return self.env.sim.data.body_xpos[body_id].copy()

    def _is_object_grasped(self, target: PlacementTarget) -> bool:
        """Return whether both gripper fingers currently contact the target object."""
        object_model = next(
            object_model
            for object_model in self.env.objects
            if object_model.name == target.class_name
        )
        return bool(
            self.env._check_grasp(
                gripper=self.env.robots[0].gripper,
                object_geoms=object_model.contact_geoms,
            )
        )

    def _object_lifted(self, target: PlacementTarget, start_z: float) -> bool:
        """Return whether the target remains grasped and rose above its start height."""
        current_z = self._object_position(target)[2]
        return self._is_object_grasped(target) and current_z >= (
            start_z + self.stages_config.grasp_min_lift_height
        )

    @staticmethod
    def _yaw_deg(orientation: np.ndarray) -> float:
        """Return symmetry-wrapped world yaw from an orientation matrix."""
        return CoordinateTransformer.rotation_matrix_to_yaw_deg(orientation)

    def _yaw_error(self, target_yaw: float, orientation: np.ndarray) -> float:
        """Return the shortest symmetry-aware yaw error in degrees."""
        current_yaw = self._yaw_deg(orientation)
        return (target_yaw - current_yaw + 90.0) % 180.0 - 90.0

    @staticmethod
    def _required_rezero_yaw(state: ControllerRunState) -> float:
        """Return the recorded re-zero yaw or raise if it is unavailable."""
        if state.rezero_yaw_deg is None:
            raise RuntimeError("MOVE_TO_BIN requires a yaw recorded at REZERO_YAW")
        return state.rezero_yaw_deg

    @staticmethod
    def _required_rezero_orientation(state: ControllerRunState) -> np.ndarray:
        """Return the recorded re-zero orientation or raise if unavailable."""
        if state.rezero_orientation is None:
            raise RuntimeError(
                "MOVE_TO_BIN requires an orientation recorded at REZERO_YAW"
            )
        return state.rezero_orientation
