import logging
import time
from pathlib import Path
from typing import Sequence

import cv2
import mujoco
import numpy as np
import robosuite as suite
from robosuite.controllers import load_composite_controller_config
from robosuite.environments.base import MujocoEnv

from src.control.pickplace_controller import PickPlaceController
from src.control.stage import Stage
from src.environments.pickplace_with_robot_offset import PickPlaceWithRobotOffset
from src.geometry.coordinate_transform import CoordinateTransformer
from src.recording.run_observer import BaseRunObserver
from src.util.types import (
    Config,
    CropRegion,
    Detection,
    ImageSize,
    PlacementTarget,
    RunResult,
)
from src.vision.onnx_detector import OnnxDetector
from src.vision.pose_estimator import PoseEstimator


class PickPlaceExecutor:
    """Coordinate one simulated vision-guided pick-and-place episode.

    Before every placement, the executor captures the current agent-view image,
    detects all visible configured objects, and estimates their poses. Then it
    selects only the next required class. Scanning again after a release lets
    objects that were hidden behind another object become available.

    Camera frames come from a direct MuJoCo render rather than Robosuite's own
    camera pipeline (see ``_init_env`` and ``_render_camera_frame``): Robosuite's
    offscreen path re-renders a full agentview frame on every ``env.step()``,
    but this executor only needs one frame per detection scan, and offscreen
    rendering also conflicts with the interactive viewer on macOS.
    """

    def __init__(
        self,
        config: Config,
        detector: OnnxDetector,
        pose_estimator: PoseEstimator,
        image_size: ImageSize,
        crop_region: CropRegion,
        has_renderer: bool = True,
        controller_factory=None,
        logger=None,
        pause_flag_path: Path | None = None,
        observers: Sequence[BaseRunObserver] = (),
    ) -> None:
        """Initialize the environment, perception pipeline, and controller.

        Args:
            config: Typed application configuration.
            detector: Object detector used on agent-view images.
            pose_estimator: Pose estimator used for detected objects.
            image_size: Rendered camera-frame dimensions.
            crop_region: Image region supplied to vision models.
            has_renderer: Whether to open the interactive MuJoCo viewer.
            controller_factory: Optional factory for injecting a controller.
            logger: Logger used for pipeline progress and diagnostics.
            pause_flag_path: Path of a flag file that pauses the run. After
                every controller step the executor checks this path: as long
                as a file exists there, the simulation waits before stepping
                again. ``None`` means the run cannot be paused.
            observers: Callbacks that watch the run (recorders, visualizers).
                See :class:`BaseRunObserver` for the hooks.
        """
        self.config = config
        self.detector = detector
        self.pose_estimator = pose_estimator
        self.image_size = image_size
        self.crop_region = crop_region
        self.target_classes = config.execution.target_classes
        self.logger = logger or logging.getLogger(__name__)
        self._pause_flag_path = pause_flag_path
        self._observers = list(observers)

        # One mujoco.Renderer per camera, created lazily on first use.
        # geomgroup[0] = 0 hides collision geometry, matching what the
        # interactive viewer shows.
        self._renderers: dict[str, mujoco.Renderer] = {}
        self._scene_option = mujoco.MjvOption()
        self._scene_option.geomgroup[0] = 0

        # Per-run measurements of the vision models against the simulator's
        # ground truth. Aggregated into the RunResult that run() returns.
        self._detection_scan_index = 0
        self._detection_confidences: list[float] = []
        self._pose_position_errors_m: list[float] = []
        self._pose_rotation_errors_deg: list[float | None] = []

        # 180-degree local symmetry about the object's own up-axis; see
        # _rotation_error_deg for why this is needed.
        self.symmetry_local_rotations = (
            np.eye(3),
            np.diag([-1.0, -1.0, 1.0]),
        )

        self.env = self._init_env(has_renderer)
        self.cam_xpos, self.cam_xmat = self._resolve_camera_extrinsics("agentview")

        build_controller = controller_factory or (
            lambda: PickPlaceController(
                self.env,
                config.motion,
                config.stages,
                frame_callback=self._on_controller_step,
            )
        )
        self.controller = build_controller()

        self.logger.info(
            "PickPlaceExecutor initialized | controller=%s",
            type(self.controller).__name__,
        )

    # ------------------------------------------------------------------
    # Environment
    # ------------------------------------------------------------------

    def _init_env(self, has_renderer: bool) -> MujocoEnv:
        """Create the Panda PickPlace simulation used by this application.

        ``has_offscreen_renderer`` and ``use_camera_obs`` are left off (see the
        class docstring for why frames are instead pulled via a direct MuJoCo
        render). ``horizon`` is set far above any single pick because one
        episode here must run every configured placement back to back, not
        just one.

        Args:
            has_renderer: Whether to open the interactive MuJoCo viewer.

        Returns:
            Robosuite environment with a Panda, Robotiq85 gripper, and agent view.
        """
        controller_config = load_composite_controller_config(
            controller=str(Path("config", "joint_position_controller.json"))
        )

        env = suite.make(
            PickPlaceWithRobotOffset.__name__,
            robots="Panda",
            controller_configs=controller_config,
            has_renderer=has_renderer,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            render_camera=None,
            camera_names="agentview",
            camera_heights=self.image_size.height,
            camera_widths=self.image_size.width,
            gripper_types="Robotiq85Gripper",
            initialization_noise=None,
            hard_reset=False,
            seed=self.config.environment.seed,
            robot_base_offset=self.config.environment.robot_base_offset,
            horizon=20000,
        )
        self._boost_gripper_clamp_force(env)
        return env

    def _boost_gripper_clamp_force(self, env) -> None:
        """Apply the configured multiplier to both Robotiq85 finger actuators.

        Args:
            env: Robosuite environment whose gripper finger actuators are
                scaled in place.
        """
        multiplier = self.config.motion.gripper_clamp_force_multiplier
        model = env.sim.model._model
        for name in ("gripper0_right_finger_1", "gripper0_right_finger_2"):
            actuator_id = env.sim.model.actuator_name2id(name)
            model.actuator_gainprm[actuator_id][0] *= multiplier
            model.actuator_biasprm[actuator_id][1] *= multiplier

    def _resolve_camera_extrinsics(self, camera_name: str):
        """Return the world-frame position and orientation of ``camera_name``.

        Args:
            camera_name: Name of the MuJoCo camera to resolve.
        """
        cam_id = self.env.sim.model.camera_name2id(camera_name)
        cam_xpos = self.env.sim.data.cam_xpos[cam_id].copy()
        cam_xmat = self.env.sim.data.cam_xmat[cam_id].copy().reshape(3, 3)
        return cam_xpos, cam_xmat

    # ------------------------------------------------------------------
    # Controller callback
    # ------------------------------------------------------------------

    def _on_controller_step(self, object_name: str, stage: Stage) -> None:
        """Runs after every controller step: pause first, then notify observers.

        Args:
            object_name: Class name of the object the controller is currently
                acting on.
            stage: Stage of the pick-and-place motion the controller is
                currently executing.
        """
        self._wait_while_paused()
        for observer in self._observers:
            observer.on_step(object_name, stage)

    def _wait_while_paused(self) -> None:
        """Block between ticks while an external pause flag file exists."""
        if self._pause_flag_path is None:
            return
        while self._pause_flag_path.exists():
            time.sleep(0.2)

    # ------------------------------------------------------------------
    # Perception
    # ------------------------------------------------------------------

    def _render_camera_frame(self, camera_name: str) -> np.ndarray:
        """Render a camera directly through MuJoCo.

        Args:
            camera_name: Name of the MuJoCo camera to render.
        """
        if camera_name not in self._renderers:
            # Direct rendering does not inherit Robosuite's camera buffer size.
            model = self.env.sim.model._model
            model.vis.global_.offwidth = max(
                model.vis.global_.offwidth, self.image_size.width
            )
            model.vis.global_.offheight = max(
                model.vis.global_.offheight, self.image_size.height
            )
            self._renderers[camera_name] = mujoco.Renderer(
                model,
                height=self.image_size.height,
                width=self.image_size.width,
            )

        renderer = self._renderers[camera_name]
        renderer.update_scene(
            self.env.sim.data._data, camera=camera_name, scene_option=self._scene_option
        )
        return renderer.render()

    def _capture_agentview_frame(self) -> np.ndarray:
        """Capture a full BGR agent-view image without detector cropping."""
        image = self._render_camera_frame("agentview")
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    @staticmethod
    def _keep_best_per_class(detections: list[Detection]) -> list[Detection]:
        """Keeps only the highest-confidence detection per class.

        Args:
            detections: Raw detections from one detector call, possibly with
                multiple detections for the same class.
        """

        best_by_class: dict[int, Detection] = {}
        for detection in detections:
            current_best = best_by_class.get(detection.class_id)
            if current_best is None or detection.confidence > current_best.confidence:
                best_by_class[detection.class_id] = detection
        return list(best_by_class.values())

    def _detect_targets(self, sought_class_name: str) -> list[PlacementTarget]:
        """Find visible configured objects and convert their poses to world coordinates.

        Detection boxes are predicted in cropped agent-view pixels. The pose
        model returns camera-frame position and rotation, which are transformed
        into the simulation world frame before the controller uses them.

        Args:
            sought_class_name: The object class this scan is looking for. Other
                configured classes visible in the same frame are detected too,
                but only this one's confidence and pose error are recorded.
                Otherwise, an object that stays in view across several scans
                before its own turn would be double-counted in the run stats.
        """
        self._detection_scan_index += 1
        agentview_frame = self._capture_agentview_frame()
        agentview_image = agentview_frame[
            self.crop_region.y1 : self.crop_region.y2,
            self.crop_region.x1 : self.crop_region.x2,
        ]

        detections = self.detector.predict(
            agentview_image,
            conf_threshold=self.config.detector.conf_threshold,
            iou_threshold=self.config.detector.iou_threshold,
        )
        detections = self._keep_best_per_class(detections)
        self.logger.info(
            "Scan %d | detected=%d", self._detection_scan_index, len(detections)
        )

        targets = []
        poses_cam = []
        for detection in detections:
            class_name = self.config.detector.class_names[detection.class_id]
            if class_name not in self.target_classes:
                continue

            xyz_cam, rot_cam = self.pose_estimator.predict(
                agentview_image, detection.box, detection.class_id
            )
            poses_cam.append((class_name, detection.box, xyz_cam, rot_cam))
            world_xpos, world_xmat, yaw_deg = (
                CoordinateTransformer.camera_to_world_frame(
                    xyz_cam, rot_cam, self.cam_xpos, self.cam_xmat
                )
            )

            position_error_m, rotation_error_deg = self._pose_errors(
                class_name, world_xpos, world_xmat
            )
            if class_name == sought_class_name:
                self._detection_confidences.append(float(detection.confidence))
                self._pose_position_errors_m.append(position_error_m)
                self._pose_rotation_errors_deg.append(rotation_error_deg)
                for observer in self._observers:
                    observer.on_target_detected(class_name, poses_cam[-1])

            bin_index = self.env.object_to_id[class_name.lower()]
            targets.append(
                PlacementTarget(
                    class_name=class_name,
                    object_pos=world_xpos,
                    yaw_deg=yaw_deg,
                    bin_pos=self.env.target_bin_placements[bin_index],
                )
            )

            rotation_display = (
                "n/a (rotation-symmetric)"
                if rotation_error_deg is None
                else f"{rotation_error_deg:.1f}°"
            )
            self.logger.info(
                "Scan %d | %s detected | confidence=%.2f | "
                "world_position=(%.3f, %.3f, %.3f) | grasp_yaw=%.1f° | "
                "position_error=%.1fcm | rotation_error=%s",
                self._detection_scan_index,
                class_name,
                detection.confidence,
                world_xpos[0],
                world_xpos[1],
                world_xpos[2],
                yaw_deg,
                position_error_m * 100,
                rotation_display,
            )

        targets.sort(key=self._pick_order_key)
        for observer in self._observers:
            observer.on_scan(agentview_frame, poses_cam)
        return targets

    def _pick_order_key(self, target: PlacementTarget) -> int:
        """Return the configured placement-order index for ``target``.

        Args:
            target: Detected placement target whose class determines its
                position in ``target_classes``.
        """
        try:
            return self.target_classes.index(target.class_name)
        except ValueError:
            return len(self.target_classes)

    # ------------------------------------------------------------------
    # Vision accuracy against simulator ground truth
    # ------------------------------------------------------------------

    def _pose_errors(
        self, class_name: str, world_xpos: np.ndarray, world_xmat: np.ndarray
    ) -> tuple[float, float | None]:
        """Compare a predicted world pose against the simulator's true pose.

        Args:
            class_name: Object class whose predicted pose is being checked;
                used to look up the simulator's ground-truth body pose.
            world_xpos: Predicted world-frame position of the object.
            world_xmat: Predicted world-frame rotation matrix of the object.

        Returns:
            Position error in metres, and rotation error in degrees. The
            rotation error is ``None`` for classes in
            ``PoseConfig.rotation_symmetric_classes`` (e.g. a can), since
            they have no meaningful "correct" rotation to compare against.
        """
        body_id = self.env.obj_body_id[class_name]
        true_pos = self.env.sim.data.body_xpos[body_id]
        position_error_m = float(np.linalg.norm(world_xpos - true_pos))

        if class_name in self.config.pose.rotation_symmetric_classes:
            return position_error_m, None

        true_mat = self.env.sim.data.body_xmat[body_id].reshape(3, 3)
        return position_error_m, self._rotation_error_deg(world_xmat, true_mat)

    def _rotation_error_deg(
        self, rot_pred_world: np.ndarray, rot_target_world: np.ndarray
    ) -> float:
        """Symmetry-aware geodesic angle between two rotations, in degrees.

        Takes the minimum angle over ``self.symmetry_local_rotations`` so a
        180-degree rotation about an object's own up-axis, which looks
        identical for these objects, isn't counted as an error just because
        it doesn't match the simulator's arbitrarily-chosen orientation. The
        same metric the vision repo reports for the pose estimator.

        Args:
            rot_pred_world: Predicted world-frame rotation matrix.
            rot_target_world: Simulator's ground-truth world-frame rotation
                matrix.
        """
        angles = []
        for symmetry in self.symmetry_local_rotations:
            relative = rot_pred_world.T @ (rot_target_world @ symmetry)
            cos_angle = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
            angles.append(float(np.degrees(np.arccos(cos_angle))))
        return min(angles)

    @staticmethod
    def _average(values: list[float]) -> float | None:
        """Return the mean of ``values``, or ``None`` if it's empty.

        Args:
            values: Numbers to average, e.g. per-object error or confidence
                measurements collected over a run.
        """
        return sum(values) / len(values) if values else None

    def _build_run_result(self, placed: int) -> RunResult:
        """Aggregate the per-object stats collected during the run.

        Args:
            placed: Number of objects successfully placed before the run
                stopped or completed.
        """

        position_errors_cm = [error * 100 for error in self._pose_position_errors_m]
        # Rotation-symmetric classes (e.g. Can) contribute None here, since
        # they have no meaningful "correct" rotation. Excluded from the
        # average so they can't drag it down with meaningless noise.
        scored_rotation_errors = [
            error for error in self._pose_rotation_errors_deg if error is not None
        ]
        return RunResult(
            placed=placed,
            avg_detection_confidence=self._average(self._detection_confidences),
            avg_pose_position_error_cm=self._average(position_errors_cm),
            avg_pose_rotation_error_deg=self._average(scored_rotation_errors),
            detection_confidences=list(self._detection_confidences),
            pose_position_errors_cm=position_errors_cm,
            pose_rotation_errors_deg=list(self._pose_rotation_errors_deg),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> RunResult:
        """Reset the scene and place visible objects in the configured order.

        The run stops if the next required object is not visible or a grasp
        fails. This preserves the requested order instead of silently picking
        a different object.

        Returns:
            Number of objects placed, plus this run's vision-accuracy stats.
        """
        obs = self.env.reset()
        self.logger.info(
            "Run started | seed=%s | pick_order=%s",
            self.config.environment.seed,
            ", ".join(self.target_classes),
        )
        for observer in self._observers:
            observer.on_run_start(self)

        placed_classes: list[str] = []
        start_simulation_time = float(self.env.sim.data.time)
        result: RunResult

        try:
            for index, class_name in enumerate(self.target_classes):
                targets = self._detect_targets(sought_class_name=class_name)
                target = next(
                    (
                        candidate
                        for candidate in targets
                        if candidate.class_name == class_name
                    ),
                    None,
                )
                if target is None:
                    self.logger.warning(
                        "Run stopped | required object %s is not visible | "
                        "completed=%d/%d",
                        class_name,
                        len(placed_classes),
                        len(self.target_classes),
                    )
                    break

                obs, success = self.controller.run(
                    obs,
                    target,
                    is_last=index == len(self.target_classes) - 1,
                    reset_osc=index == 0,
                )
                if not success:
                    self.logger.error(
                        "Run stopped | %s was not successfully grasped or lifted | "
                        "completed=%d/%d",
                        class_name,
                        len(placed_classes),
                        len(self.target_classes),
                    )
                    break
                placed_classes.append(class_name)
        finally:
            result = self._build_run_result(placed=len(placed_classes))
            for observer in self._observers:
                observer.on_run_end(result)

        simulation_duration = float(self.env.sim.data.time) - start_simulation_time
        self.logger.info(
            "SUMMARY | objects_placed=%d/%d | placed=%s | simulation_duration=%.1f s",
            len(placed_classes),
            len(self.target_classes),
            ", ".join(placed_classes) or "none",
            simulation_duration,
        )
        return result

    def close(self) -> None:
        """Close direct renderers and the Robosuite environment."""
        for renderer in self._renderers.values():
            renderer.close()
        self.env.close()
