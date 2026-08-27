"""Connect the simulation, vision models, and controller for a complete run.

The executor owns one Robosuite episode. Per configured object it captures an
agent-view image, detects the object, estimates its pose, converts that to
world coordinates, and asks the controller to pick and place it.

Video recording, trajectory recording, and frame annotation live outside the
executor as :class:`~src.recording.run_observer.BaseRunObserver` implementations
that watch the run through its lifecycle hooks.
"""

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
    detects all visible configured objects, estimates their poses, and selects
    only the next required class. Scanning again after a release lets objects
    that were hidden behind another object become available.

    Attributes:
        config: Typed application configuration.
        detector: ONNX detector for agent-view images.
        pose_estimator: ONNX pose estimator for detected objects.
        env: Initialized Robosuite PickPlace environment.
        controller: Controller that executes one object placement.
        target_classes: Required object order for the current episode.
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
            observers: Callbacks that watch the run (recorders, visualizers);
                see :class:`BaseRunObserver` for the hooks.
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

        # Direct MuJoCo capture instead of Robosuite's camera pipeline: one
        # lazily created renderer per camera, with collision geometry hidden
        # to match what the interactive viewer shows.
        self._renderers: dict[str, mujoco.Renderer] = {}
        self._scene_option = mujoco.MjvOption()
        self._scene_option.geomgroup[0] = 0

        # Per-run measurement of the vision models against simulator ground
        # truth; aggregated into the RunResult that run() returns.
        self._detection_scan_index = 0
        self._detection_confidences: list[float] = []
        self._pose_position_errors_m: list[float] = []
        self._pose_rotation_errors_deg: list[float] = []

        # 180-degree local symmetry: without it, objects that look identical
        # rotated 180 degrees about their own up-axis would be penalized for
        # visually correct rotations.
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

        Returns:
            Robosuite environment with a Panda, Robotiq85 gripper, and agent view.
        """
        controller_config = load_composite_controller_config(
            controller=str(Path("config", "joint_position_controller.json"))
        )

        # Robosuite's own camera pipeline stays off (has_offscreen_renderer /
        # use_camera_obs): with use_camera_obs=True it renders a full agentview
        # frame on every single env.step() while this executor only needs one
        # frame per detection scan (see _render_camera_frame). Off-screen
        # rendering also conflicts with the interactive viewer on macOS.
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
            # One episode contains all configured placements.
            horizon=20000,
        )
        self._boost_gripper_clamp_force(env)
        return env

    def _boost_gripper_clamp_force(self, env) -> None:
        """Apply the configured multiplier to both Robotiq85 finger actuators."""
        multiplier = self.config.motion.gripper_clamp_force_multiplier
        model = env.sim.model._model
        for name in ("gripper0_right_finger_1", "gripper0_right_finger_2"):
            actuator_id = env.sim.model.actuator_name2id(name)
            model.actuator_gainprm[actuator_id][0] *= multiplier
            model.actuator_biasprm[actuator_id][1] *= multiplier

    def _resolve_camera_extrinsics(self, camera_name: str):
        """Return the world-frame position and orientation of ``camera_name``."""
        cam_id = self.env.sim.model.camera_name2id(camera_name)
        cam_xpos = self.env.sim.data.cam_xpos[cam_id].copy()
        cam_xmat = self.env.sim.data.cam_xmat[cam_id].copy().reshape(3, 3)
        return cam_xpos, cam_xmat

    # ------------------------------------------------------------------
    # Controller callback
    # ------------------------------------------------------------------

    def _on_controller_step(self, object_name: str, stage: Stage) -> None:
        """Runs after every controller step: pause first, then notify observers."""
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
        """Render a camera directly through MuJoCo."""
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
        """Keeps only the highest-confidence detection per class."""
        
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
                but only this one's confidence/pose-error is recorded --
                otherwise an object sitting in view across several scans before
                its own turn would be double-counted in the run stats.
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

            self.logger.info(
                "Scan %d | %s detected | confidence=%.2f | "
                "world_position=(%.3f, %.3f, %.3f) | grasp_yaw=%.1f° | "
                "position_error=%.1fcm | rotation_error=%.1f°",
                self._detection_scan_index,
                class_name,
                detection.confidence,
                world_xpos[0],
                world_xpos[1],
                world_xpos[2],
                yaw_deg,
                position_error_m * 100,
                rotation_error_deg,
            )

        targets.sort(key=self._pick_order_key)
        for observer in self._observers:
            observer.on_scan(agentview_frame, poses_cam)
        return targets

    def _pick_order_key(self, target: PlacementTarget) -> int:
        """Return the configured placement-order index for ``target``."""
        try:
            return self.target_classes.index(target.class_name)
        except ValueError:
            return len(self.target_classes)

    # ------------------------------------------------------------------
    # Vision accuracy against simulator ground truth
    # ------------------------------------------------------------------

    def _pose_errors(
        self, class_name: str, world_xpos: np.ndarray, world_xmat: np.ndarray
    ) -> tuple[float, float]:
        """Compare a predicted world pose against the simulator's true pose.

        Returns:
            Position error in metres and rotation error in degrees.
        """
        body_id = self.env.obj_body_id[class_name]
        true_pos = self.env.sim.data.body_xpos[body_id]
        true_mat = self.env.sim.data.body_xmat[body_id].reshape(3, 3)
        return (
            float(np.linalg.norm(world_xpos - true_pos)),
            self._rotation_error_deg(world_xmat, true_mat),
        )

    def _rotation_error_deg(
        self, rot_pred_world: np.ndarray, rot_target_world: np.ndarray
    ) -> float:
        """Symmetry-aware geodesic angle between two rotations, in degrees.

        The same metric the vision repo reports for the pose estimator.
        """
        angles = []
        for symmetry in self.symmetry_local_rotations:
            relative = rot_pred_world.T @ (rot_target_world @ symmetry)
            cos_angle = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
            angles.append(float(np.degrees(np.arccos(cos_angle))))
        return min(angles)

    def _build_run_result(self, placed: int) -> RunResult:
        """Aggregate the per-object stats collected during the run."""

        def average(values: list[float]) -> float | None:
            return sum(values) / len(values) if values else None

        position_errors_cm = [error * 100 for error in self._pose_position_errors_m]
        return RunResult(
            placed=placed,
            avg_detection_confidence=average(self._detection_confidences),
            avg_pose_position_error_cm=average(position_errors_cm),
            avg_pose_rotation_error_deg=average(self._pose_rotation_errors_deg),
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
            # Also after a crash, so recorders can still write what they have.
            for observer in self._observers:
                observer.on_run_end()

        simulation_duration = float(self.env.sim.data.time) - start_simulation_time
        self.logger.info(
            "SUMMARY | objects_placed=%d/%d | placed=%s | simulation_duration=%.1f s",
            len(placed_classes),
            len(self.target_classes),
            ", ".join(placed_classes) or "none",
            simulation_duration,
        )
        return self._build_run_result(placed=len(placed_classes))

    def close(self) -> None:
        """Close direct renderers and the Robosuite environment."""
        for renderer in self._renderers.values():
            renderer.close()
        self.env.close()
