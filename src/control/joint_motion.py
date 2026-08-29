import mujoco
import numpy as np
from robosuite.utils.control_utils import orientation_error
from scipy.optimize import least_squares, minimize_scalar

from src.util.types import MotionConfig, WristTargets


class JointMotion:
    """Convert Cartesian pick-and-place goals into absolute Panda joint targets.

    The class reads the current MuJoCo pose, takes one small differential-IK
    step toward the requested gripper pose, and separately tracks J5/J6 to
    keep the gripper pointing down. It does not teleport the robot to a goal:
    every returned action is limited by the configured per-tick motion bounds.

    J5/J6 are deliberately excluded from the IK Jacobian and driven instead
    by a dedicated proportional-plus-feedforward tracker: letting IK's
    orientation term and the wrist tracker both act on the same roll/pitch
    DOF would have them fight over it. This is also why
    ``MotionConfig.orientation_weight`` defaults to 0: IK stays
    position-only unless a stage explicitly asks for world-yaw tracking via
    ``keep_world_yaw``.
    """

    def __init__(self, env, config: MotionConfig) -> None:
        """Initialize MuJoCo references used for differential IK.

        Args:
            env: Robosuite environment containing the Panda robot.
            config: Motion gains, limits, and joint names.
        """
        self.env = env
        self.config = config
        robot = env.robots[0]

        self.joint_qpos_ids = np.asarray(robot._ref_joint_pos_indexes, dtype=int)
        self.joint_dof_ids = np.asarray(robot._ref_joint_vel_indexes, dtype=int)
        self.joint_model_ids = np.asarray(robot._ref_joint_indexes, dtype=int)
        self.joint5_index = self._joint_index(config.joint5_name)
        self.joint6_index = self._joint_index(config.joint6_name)
        self.eef_site_id = env.sim.model.site_name2id("gripper0_right_grip_site")

        # Wrist joints are controlled by vertical-down tracking, not by IK.
        excluded = [self.joint6_index]
        if self.joint5_index is not None:
            excluded.append(self.joint5_index)
        self.free_joint_indices = np.delete(
            np.arange(len(self.joint_dof_ids)), excluded
        )

        self.joint6_target = 0.0
        self._previous_wrist_targets: WristTargets | None = None

    def _joint_index(self, name: str | None) -> int | None:
        """Return an action-space joint index for a Robosuite joint name.

        Args:
            name: Robosuite joint name, or ``None`` if the joint isn't
                configured.
        """
        return (
            None if name is None else self.env.robots[0].robot_model.joints.index(name)
        )

    @property
    def joint_positions(self) -> np.ndarray:
        """Return a copy of the arm's current joint positions in radians."""
        return self.env.sim.data.qpos[self.joint_qpos_ids].copy()

    def eef_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the end-effector world position and rotation matrix."""
        return (
            self.env.sim.data.site_xpos[self.eef_site_id].copy(),
            self.env.sim.data.site_xmat[self.eef_site_id].reshape(3, 3).copy(),
        )

    def reset_posture(self) -> None:
        """Reset the wrist controller before starting a new object.

        Joint 6 starts from its current angle unless ``JOINT6_HOME_DEG`` is
        configured. Clearing the previous target prevents feedforward from a
        prior object's final wrist pose affecting the next object.
        """
        home_angle = self.config.joint6_home_deg
        self.joint6_target = (
            np.deg2rad(home_angle)
            if home_angle is not None
            else self.joint_positions[self.joint6_index]
        )
        self._previous_wrist_targets = None

    def downward_joint6_target(self) -> float:
        """Find the Joint-6 angle that makes the gripper point downward.

        This is used when J5 is unavailable. It tests allowed Joint-6 angles
        using forward kinematics and picks the one with the smallest tool
        tilt from straight down (negative world Z). There is no closed form
        for this angle: with the rest of the arm fixed, tilt is a
        nonlinear function of J6, so it is found by a bounded 1-D search
        instead.
        """

        current_joints = self.joint_positions
        lower, upper = self._joint_limits(self.joint6_index)

        try:
            result = minimize_scalar(
                self._joint6_tilt_error,
                args=(current_joints,),
                bounds=(lower, upper),
                method="bounded",
                options={"xatol": np.deg2rad(0.01)},
            )
            return float(result.x)
        finally:
            self._set_joint_positions(current_joints)

    def _joint6_tilt_error(self, angle: float, current_joints: np.ndarray) -> float:
        """Return vertical-down tilt error for a candidate Joint-6 angle.

        Args:
            angle: Candidate Joint-6 angle to test, in radians.
            current_joints: Full joint-position vector to vary Joint-6 within.
        """
        candidate = current_joints.copy()
        candidate[self.joint6_index] = angle
        # Test candidate forward kinematics without stepping physics.
        self._set_joint_positions(candidate)
        return 1.0 + self.eef_pose()[1][2, 2]

    def vertical_down_wrist_targets(self) -> WristTargets:
        """Find J5/J6 targets that make the gripper point vertically down.

        The solver keeps the rest of the arm fixed and changes only the wrist.
        This gives the Cartesian IK solver a stable orientation constraint while
        the robot raises, aligns over an object, descends, or moves to a bin.
        """

        if self.joint5_index is None:
            return WristTargets(None, self.downward_joint6_target())

        current_joints = self.joint_positions
        joint_indices = np.array([self.joint5_index, self.joint6_index])
        lower = np.array([self._joint_limits(index)[0] for index in joint_indices])
        upper = np.array([self._joint_limits(index)[1] for index in joint_indices])

        try:
            result = least_squares(
                self._horizontal_tool_axis,
                x0=current_joints[joint_indices],
                args=(current_joints, joint_indices),
                bounds=(lower, upper),
                xtol=1e-10,
                ftol=1e-10,
                gtol=1e-10,
            )
            return WristTargets(float(result.x[0]), float(result.x[1]))
        finally:
            self._set_joint_positions(current_joints)

    def _horizontal_tool_axis(
        self,
        angles: np.ndarray,
        current_joints: np.ndarray,
        joint_indices: np.ndarray,
    ) -> np.ndarray:
        """Return the horizontal components of tool Z for candidate wrist angles.

        Args:
            angles: Candidate J5/J6 angles to test, in radians.
            current_joints: Full joint-position vector to vary J5/J6 within.
            joint_indices: Indices of J5 and J6 within the joint-position vector.
        """
        candidate = current_joints.copy()
        candidate[joint_indices] = angles
        self._set_joint_positions(candidate)
        return self.eef_pose()[1][:2, 2]

    def downward_joint6_diagnostics(self) -> tuple[float, float, float]:
        """Return Joint-6 target, actual tilt, and predicted target tilt in degrees."""

        current_joints = self.joint_positions
        actual_tilt = self._tool_tilt_deg()
        target = self.downward_joint6_target()
        candidate = current_joints.copy()
        candidate[self.joint6_index] = target
        try:
            self._set_joint_positions(candidate)
            return target, actual_tilt, self._tool_tilt_deg()
        finally:
            self._set_joint_positions(current_joints)

    def vertical_down_joint5_joint6_diagnostics(self) -> tuple[float, float, float]:
        """Compatibility helper used by the linear diagnostic."""

        targets = self.vertical_down_wrist_targets()
        if targets.joint5 is None:
            raise ValueError("Joint-5 diagnostics require MOTION.JOINT5_NAME")

        current_joints = self.joint_positions
        candidate = current_joints.copy()
        candidate[self.joint5_index] = targets.joint5
        candidate[self.joint6_index] = targets.joint6
        try:
            self._set_joint_positions(candidate)
            return targets.joint5, targets.joint6, self._tool_tilt_deg()
        finally:
            self._set_joint_positions(current_joints)

    def joint7_yaw_action(
        self, yaw_error_deg: float, gripper_action: float
    ) -> np.ndarray:
        """Return one bounded Joint-7 action for a gripper-yaw correction.

        Used by yaw-only stages instead of routing the correction through
        differential IK. With the wrist held vertical-down by J5/J6, J7 (the
        last joint before the gripper) rotates the tool about its own axis,
        so changing it adjusts yaw without moving the end effector.
        All other arm joints remain at their measured positions so yaw-only
        stages cannot accidentally move the gripper over the object.

        Args:
            yaw_error_deg: Signed yaw error to correct, in degrees.
            gripper_action: Gripper command appended to the returned action.
        """

        joint_targets = self.joint_positions
        step_deg = np.clip(
            -yaw_error_deg,
            -self.config.yaw_max_step_deg,
            self.config.yaw_max_step_deg,
        )
        joint_targets[self.config.yaw_joint_index] += np.deg2rad(step_deg)
        return np.concatenate((joint_targets, [gripper_action]))

    def action(
        self,
        target_position: np.ndarray,
        target_orientation: np.ndarray,
        gripper_action: float,
        orientation_weight: float | None = None,
        joint6_target: float | None = None,
        joint5_target: float | None = None,
        joint7_target: float | None = None,
        position_scale: float = 1.0,
        max_position_delta: float | None = None,
        rotation_axis_weights: np.ndarray | None = None,
    ) -> np.ndarray:
        """Build one safe arm action that advances toward a Cartesian pose.

        Combines one damped-least-squares IK step (position, and orientation
        only when ``orientation_weight``/``rotation_axis_weights`` ask for
        it) with the separate J5/J6 wrist tracker and an optional direct J7
        yaw hold. IK's Jacobian columns exclude whichever joints are being
        driven by tracking instead, so the two mechanisms never both try to
        move the same joint. Every component is clamped before being
        combined, so an unreachable target or a near-singular pose produces
        a bounded step rather than a lurch.

        Args:
            target_position: Desired end-effector world position in metres.
            target_orientation: Desired end-effector world rotation matrix.
            gripper_action: Gripper command appended to the arm action.
            orientation_weight: Weight applied to orientation correction.
            joint6_target: Optional Joint-6 target in radians.
            joint5_target: Optional Joint-5 target in radians.
            joint7_target: Optional Joint-7 target in radians.
            position_scale: Multiplier applied to Cartesian position error.
            max_position_delta: Per-axis Cartesian step limit in metres.
            rotation_axis_weights: Per-axis weights for rotation correction.

        Returns:
            Seven absolute Panda joint targets followed by the gripper action.
        """

        current_joints = self.joint_positions
        current_position, current_orientation = self.eef_pose()
        joint6_target = self.joint6_target if joint6_target is None else joint6_target
        orientation_weight = (
            self.config.orientation_weight
            if orientation_weight is None
            else orientation_weight
        )
        axis_weights = self._rotation_axis_weights(rotation_axis_weights)
        ik_indices = self._ik_joint_indices(joint5_target, joint7_target)

        position_delta = self._position_delta(
            position_scale * (target_position - current_position),
            max_position_delta,
        )
        rotation_delta = np.clip(
            orientation_weight
            * self.config.orientation_kp
            * axis_weights
            * orientation_error(target_orientation, current_orientation),
            -self.config.max_orientation_delta,
            self.config.max_orientation_delta,
        )
        joint_delta = np.zeros(len(current_joints))
        joint_delta[ik_indices] = self._ik_delta(
            np.concatenate((position_delta, rotation_delta)),
            ik_indices,
            orientation_weight * axis_weights,
        )

        self._apply_wrist_targets(
            joint_delta,
            current_joints,
            WristTargets(joint5_target, joint6_target),
        )
        if joint7_target is not None:
            joint7_index = self.config.yaw_joint_index
            joint_delta[joint7_index] = np.clip(
                self.config.joint7_kp * (joint7_target - current_joints[joint7_index]),
                -self.config.max_joint_delta,
                self.config.max_joint_delta,
            )

        joint_targets = current_joints + joint_delta
        if self.config.lock_joint6:
            joint_targets[self.joint6_index] = joint6_target
        return np.concatenate((joint_targets, [gripper_action]))

    def _joint_limits(self, joint_index: int) -> tuple[float, float]:
        """Return lower and upper joint limits in radians for ``joint_index``.

        Args:
            joint_index: Action-space index of the joint to look up.
        """
        lower, upper = self.env.sim.model.jnt_range[self.joint_model_ids[joint_index]]
        return float(lower), float(upper)

    def _set_joint_positions(self, joints: np.ndarray) -> None:
        """Temporarily set arm joints in MuJoCo and update forward kinematics.

        Args:
            joints: Full joint-position vector to write into the simulation.
        """
        self.env.sim.data.qpos[self.joint_qpos_ids] = joints
        self.env.sim.forward()

    def _tool_tilt_deg(self, joints: np.ndarray | None = None) -> float:
        """Return end-effector tilt away from vertical down in degrees.

        Args:
            joints: Optional joint-position vector to apply first via
                forward kinematics; uses the robot's current pose if omitted.
        """
        if joints is not None:
            self._set_joint_positions(joints)
        tool_z = self.eef_pose()[1][:, 2]
        return float(np.degrees(np.arccos(np.clip(-tool_z[2], -1.0, 1.0))))

    def _rotation_axis_weights(self, weights: np.ndarray | None) -> np.ndarray:
        """Return validated per-axis rotation weights, defaulting to all axes.

        Args:
            weights: Optional per-axis (x, y, z) rotation weights.
        """
        if weights is None:
            return np.ones(3)
        weights = np.asarray(weights, dtype=float)
        if weights.shape != (3,):
            raise ValueError("rotation_axis_weights must contain exactly three values")
        return weights

    def _ik_joint_indices(
        self, joint5_target: float | None, joint7_target: float | None
    ) -> np.ndarray:
        """Select arm joints that differential IK may update this tick.

        Starts from ``free_joint_indices`` (everything except J5/J6, which
        are always driven by wrist tracking instead). J5 is added back only
        when this call has no explicit J5 target, so a caller of ``action()``
        outside the stage machine can then let IK solve for it rather than
        leaving it uncontrolled. J7 is removed whenever a yaw hold is
        requested, since that joint is being driven directly by the fixed
        ``joint7_target`` correction instead.

        Args:
            joint5_target: Optional explicit Joint-5 target for this tick.
            joint7_target: Optional explicit Joint-7 target for this tick.
        """
        indices = self.free_joint_indices
        if joint5_target is None and self.joint5_index is not None:
            indices = np.sort(np.append(indices, self.joint5_index))
        if joint7_target is not None:
            indices = indices[indices != self.config.yaw_joint_index]
        return indices

    def _position_delta(
        self,
        position_error: np.ndarray,
        max_position_delta: float | None,
    ) -> np.ndarray:
        """Scale and clamp Cartesian position error for one control tick.

        X/Y are clamped per-axis unless both exceed the limit, in which case
        the pair is rescaled by vector length instead. Clamping each axis
        independently would let a diagonal step travel up to sqrt(2) times
        farther per tick than an axis-aligned one of the same nominal speed.

        Args:
            position_error: Raw Cartesian position error, in metres.
            max_position_delta: Optional override of the configured
                per-tick cap.
        """
        delta = self.config.position_kp * position_error
        maximum = (
            self.config.max_position_delta
            if max_position_delta is None
            else max_position_delta
        )
        x, y, z = delta
        if abs(x) > maximum and abs(y) > maximum:
            length = np.hypot(x, y)
            x, y = x * maximum / length, y * maximum / length
        else:
            x, y = np.clip(x, -maximum, maximum), np.clip(y, -maximum, maximum)
        return np.array([x, y, np.clip(z, -maximum, maximum)])

    def _ik_delta(
        self,
        task_delta: np.ndarray,
        ik_indices: np.ndarray,
        rotation_row_weights: np.ndarray,
    ) -> np.ndarray:
        """Solve a damped differential-IK joint update for one task delta.

        Damping (``config.ik_damping``) keeps the solve well-conditioned
        near kinematic singularities, but a small Cartesian error can still
        map to a large joint step there. The max-component rescale
        afterward is a second, harder cap so a near-singular pose degrades
        to a slow step rather than a sudden joint jump.

        Args:
            task_delta: Concatenated position and orientation error to
                solve for.
            ik_indices: Joint indices this IK solve is allowed to update.
            rotation_row_weights: Per-axis weights applied to the rotation
                Jacobian rows before solving.
        """
        jacobian_position = np.zeros((3, self.env.sim.model.nv))
        jacobian_rotation = np.zeros((3, self.env.sim.model.nv))
        mujoco.mj_jacSite(
            self.env.sim.model._model,
            self.env.sim.data._data,
            jacobian_position,
            jacobian_rotation,
            self.eef_site_id,
        )
        jacobian = np.vstack(
            (
                jacobian_position,
                rotation_row_weights[:, None] * jacobian_rotation,
            )
        )[:, self.joint_dof_ids[ik_indices]]
        damping = self.config.ik_damping
        delta = jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + damping**2 * np.eye(6), task_delta
        )
        largest_component = np.max(np.abs(delta))
        if largest_component > self.config.max_joint_delta:
            delta *= self.config.max_joint_delta / largest_component
        return delta

    def _apply_wrist_targets(
        self,
        joint_delta: np.ndarray,
        current_joints: np.ndarray,
        targets: WristTargets,
    ) -> None:
        """Write bounded Joint-5 and Joint-6 tracking updates into ``joint_delta``.

        Args:
            joint_delta: Joint-delta array updated in place for J5/J6.
            current_joints: Robot's current joint-position vector.
            targets: Desired J5/J6 wrist targets for this tick.
        """
        previous = self._previous_wrist_targets
        if targets.joint5 is not None:
            if self.joint5_index is None:
                raise ValueError("A Joint-5 target requires MOTION.JOINT5_NAME")
            joint_delta[self.joint5_index] = self._tracking_delta(
                targets.joint5,
                None if previous is None else previous.joint5,
                current_joints[self.joint5_index],
                self.config.joint5_kp,
            )
        joint_delta[self.joint6_index] = self._tracking_delta(
            targets.joint6,
            None if previous is None else previous.joint6,
            current_joints[self.joint6_index],
            self.config.joint6_kp,
        )
        self._previous_wrist_targets = targets

    def _tracking_delta(
        self,
        target: float,
        previous_target: float | None,
        current: float,
        kp: float,
    ) -> float:
        """Return bounded proportional-plus-feedforward tracking correction.

        Feedforward is the target's own change since the last tick, added on
        top of proportional error correction. A pure-P tracker lags behind a
        target that keeps shifting: the vertical-down wrist solve is
        re-run every tick as the arm moves, so feedforward is what lets
        J5/J6 keep up with a moving target instead of trailing it.

        Args:
            target: This tick's desired joint angle, in radians.
            previous_target: Last tick's desired joint angle, or ``None``
                on the first tick after a reset.
            current: The joint's current measured angle, in radians.
            kp: Proportional gain applied to the remaining position error.
        """
        feedforward = 0.0 if previous_target is None else target - previous_target
        return float(
            np.clip(
                feedforward + kp * (target - current),
                -self.config.max_wrist_joint_delta,
                self.config.max_wrist_joint_delta,
            )
        )
