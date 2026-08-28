from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    # Only needed for type hints, so don't import it at runtime.
    from src.control.stage import Stage


@dataclass
class DetectorConfig:
    """Settings for the ONNX object detector.

    Attributes:
        model_path: Path to the detector ONNX model.
        class_names: Model class names indexed by class ID.
        image_size: Square detector input dimension in pixels.
        conf_threshold: Minimum confidence required for a detection.
        iou_threshold: Intersection-over-union threshold used by NMS.
    """

    model_path: str = "models/yolo_detector.onnx"
    class_names: List[str] = field(
        default_factory=lambda: ["Bread", "Can", "Cereal", "Milk"]
    )
    image_size: int = 640
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45


@dataclass
class PoseConfig:
    """Settings for the ONNX pose estimator.

    Attributes:
        model_path: Path to the pose-estimation ONNX model.
        pos_image_size: Full-image input size for position inference.
        rotation_image_size: Object-crop input size for rotation inference.
        rotation_symmetric_classes: Classes with no meaningful "correct"
            rotation to compare against (e.g. a cylindrical can looks and
            grasps the same at any yaw). Excluded from rotation-error
            tracking against simulator ground truth.
    """

    model_path: str = "models/pose_estimator.onnx"
    pos_image_size: int = 224
    rotation_image_size: int = 128
    rotation_symmetric_classes: List[str] = field(default_factory=list)


@dataclass
class EnvironmentConfig:
    """Simulator settings needed to reproduce a placement scene.

    Attributes:
        seed: Optional scene-layout seed. ``None`` selects a new layout.
        robot_base_offset: XYZ world-frame offset applied to the robot base.
    """

    seed: Optional[int] = 42
    robot_base_offset: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])


@dataclass
class ExecutionConfig:
    """Object order for one pick-and-place run.

    Attributes:
        target_classes: Object classes in the required placement order.
    """

    target_classes: List[str] = field(
        default_factory=lambda: ["Cereal", "Milk", "Can", "Bread"]
    )


@dataclass
class VisualizationConfig:
    """Optional image and MP4 output settings.

    Attributes:
        video_enabled: Whether to record agent-view and front-view MP4 videos.
        video_path: Base output path for the generated MP4 videos.
        video_fps: Playback speed of the output videos.
        video_capture_every_ticks: Controller steps between recorded frames.
        trajectory_enabled: Whether to export a state trajectory for web playback.
        trajectory_path: Output binary path for recorded generalized positions.
        trajectory_model_path: Reusable ZIP path for the matching MJCF model.
        scan_images_enabled: Whether to save an annotated PNG per detection scan.
        scan_images_dir: Directory the scan PNGs are written into.
    """

    video_enabled: bool = False
    video_path: str = "outputs/videos/pickplace.mp4"
    video_fps: int = 10
    video_capture_every_ticks: int = 6
    trajectory_enabled: bool = False
    trajectory_path: str = "outputs/trajectory/trajectory.bin"
    trajectory_model_path: str = "outputs/trajectory/model.zip"
    scan_images_enabled: bool = False
    scan_images_dir: str = "outputs/scans"


@dataclass
class MotionConfig:
    """Gains and per-tick limits for joint-space motion.

    Attributes:
        joint5_name: Optional Robosuite name of Joint 5.
        joint6_name: Robosuite name of Joint 6.
        yaw_joint_index: Arm-action index of the yaw joint.
        position_kp: Cartesian proportional gain used for position tracking.
        orientation_kp: Proportional gain used for orientation tracking.
        max_joint_delta: Largest arm-joint change allowed per simulation tick.
        max_position_delta: Largest Cartesian change allowed per simulation tick.
        ik_damping: Damping coefficient for differential inverse kinematics.
    """

    joint5_name: Optional[str] = "robot0_joint5"
    joint5_kp: float = 4.0
    joint5_down_tolerance_deg: float = 1.0
    joint6_name: str = "robot0_joint6"
    lock_joint6: bool = False
    joint6_kp: float = 4.0
    joint6_down_tolerance_deg: float = 1.0
    joint7_kp: float = 4.0
    yaw_joint_index: int = 6
    yaw_max_step_deg: float = 4.0
    max_joint_delta: float = 0.05
    max_wrist_joint_delta: float = 0.10
    position_kp: float = 4.0
    orientation_kp: float = 1.0
    orientation_weight: float = 0.0
    max_position_delta: float = 0.03
    max_orientation_delta: float = 0.10
    ik_damping: float = 0.03
    gripper_clamp_force_multiplier: float = 1.1
    joint6_home_deg: Optional[float] = None


@dataclass
class StagesConfig:
    """Heights, tolerances, and dwell durations for each stage.

    Attributes:
        approach_height: Shared Z height for safe horizontal travel.
        fine_descend_offsets: Per-object Z offsets for final grasp approach.
        bin_release_heights: Per-object Z heights for releases in target bins.
        position_tolerance: Cartesian convergence tolerance in metres.
        yaw_tolerance_deg: Yaw convergence tolerance in degrees.
        grasp_min_lift_height: Minimum object lift required after grasping.
        max_ticks_per_stage: Tick limit before a stage reports non-convergence.
    """

    approach_height: float = 1.2
    descend_clearance: float = 0.12
    fine_descend_offsets: Dict[str, float] = field(default_factory=dict)
    fine_descend_max_position_deltas: Dict[str, float] = field(default_factory=dict)
    bin_release_heights: Dict[str, float] = field(default_factory=dict)
    position_tolerance: float = 0.01
    bin_release_tolerance: float = 0.01
    yaw_tolerance_deg: float = 0.8
    grasp_dwell_ticks: int = 40
    grasp_min_lift_height: float = 0.03
    grasp_yaw_offset_deg: float = 0.0
    release_dwell_ticks: int = 50
    max_ticks_per_stage: int = 1500
    log_every_ticks: int = 20
    open_gripper: float = -0.05
    closed_gripper: float = 1.0


@dataclass
class Config:
    """Top-level application configuration loaded from YAML.

    Attributes:
        detector: Object-detection configuration.
        pose: Pose-estimation configuration.
        environment: Simulation configuration.
        execution: Object-order configuration.
        visualization: Optional MP4 recording configuration.
        motion: Joint-space motion configuration.
        stages: Pick-and-place stage configuration.
    """

    detector: DetectorConfig = field(default_factory=DetectorConfig)
    pose: PoseConfig = field(default_factory=PoseConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    stages: StagesConfig = field(default_factory=StagesConfig)


@dataclass
class Detection:
    """One bounding-box detection in image coordinates.

    Attributes:
        box: Bounding box as ``(x1, y1, x2, y2)`` pixels.
        confidence: Detector confidence score.
        class_id: Index into the configured detector class names.
    """

    box: Tuple[float, float, float, float]
    confidence: float
    class_id: int


@dataclass
class CropRegion:
    """Crop coordinates in image pixels, where the start is included and the end is not.

    Attributes:
        y1: Top row of the crop.
        y2: First row after the crop.
        x1: Left column of the crop.
        x2: First column after the crop.
    """

    y1: int
    y2: int
    x1: int
    x2: int


@dataclass
class ImageSize:
    """Image dimensions in pixels.

    Attributes:
        height: Image height in pixels.
        width: Image width in pixels.
    """

    height: int
    width: int


@dataclass
class PlacementTarget:
    """World-frame object pose and target-bin position.

    Attributes:
        class_name: Detected object class name.
        object_pos: XYZ object position in world coordinates.
        yaw_deg: Symmetry-wrapped world yaw in degrees.
        bin_pos: XYZ target-bin position in world coordinates.
    """

    class_name: str
    object_pos: np.ndarray  # World-frame XYZ position with shape (3,).
    yaw_deg: float  # World-frame yaw in degrees.
    bin_pos: np.ndarray  # World-frame target-bin XYZ position with shape (3,).


@dataclass
class RunResult:
    """Outcome summary for one completed pick-and-place run.

    Attributes:
        placed: Number of objects successfully placed in their target bins.
        avg_detection_confidence: Mean detector confidence (0-1) across every
            scan's detections, or ``None`` if nothing was ever detected.
        avg_pose_position_error_cm: Mean distance between the vision-estimated
            and ground-truth (simulator) object position across every scan's
            detections, in centimetres, or ``None`` if nothing was detected.
        avg_pose_rotation_error_deg: Mean symmetry-aware geodesic angle
            between the vision-estimated and ground-truth object rotation,
            in degrees, averaged over classes that aren't in
            ``PoseConfig.rotation_symmetric_classes``, or ``None`` if
            nothing scoreable was ever detected.
        detection_confidences: Per-object detector confidence (0-1), in pick
            order, so a client can track a running average as objects finish
            instead of only the final ``avg_detection_confidence``.
        pose_position_errors_cm: Per-object position error in centimetres,
            in pick order, matching ``detection_confidences`` one-for-one.
        pose_rotation_errors_deg: Per-object rotation error in degrees, in
            pick order, matching ``detection_confidences`` one-for-one.
            ``None`` for classes in ``PoseConfig.rotation_symmetric_classes``
            (e.g. a can), since they have no meaningful "correct" rotation
            to compare against.
    """

    placed: int
    avg_detection_confidence: Optional[float] = None
    avg_pose_position_error_cm: Optional[float] = None
    avg_pose_rotation_error_deg: Optional[float] = None
    detection_confidences: List[float] = field(default_factory=list)
    pose_position_errors_cm: List[float] = field(default_factory=list)
    pose_rotation_errors_deg: List[Optional[float]] = field(default_factory=list)


@dataclass
class ControllerRunState:
    """State that persists while one object moves through the stage sequence.

    Attributes:
        stage: Currently active pick-and-place stage.
        stage_tick: Number of control ticks spent in the active stage.
        grasp_joint7_target: Joint-7 position recorded after grasp yaw alignment.
        rezero_orientation: Tool orientation recorded after yaw re-zeroing.
        rezero_yaw_deg: World yaw recorded after yaw re-zeroing.
    """

    stage: Stage
    stage_tick: int = 0
    grasp_joint7_target: Optional[float] = None
    rezero_orientation: Optional[np.ndarray] = None
    rezero_yaw_deg: Optional[float] = None


@dataclass(frozen=True)
class WristTargets:
    """Desired wrist joints for a vertical-down gripper.

    Attributes:
        joint5: Optional Joint-5 target in radians.
        joint6: Joint-6 target in radians.
    """

    joint5: Optional[float]
    joint6: float


@dataclass(frozen=True)
class PoseCommand:
    """One Cartesian command for the absolute joint-position controller.

    Attributes:
        position: Target XYZ position in world coordinates.
        orientation: Target 3-by-3 world rotation matrix.
        gripper_action: Gripper command appended to the arm action.
        position_tolerance: Cartesian convergence tolerance in metres.
        joint7_target: Optional Joint-7 target in radians.
        yaw_target_deg: Optional world-yaw target in degrees.
    """

    position: np.ndarray
    orientation: np.ndarray
    gripper_action: float
    position_tolerance: float
    max_position_delta: Optional[float] = None
    joint7_target: Optional[float] = None
    orientation_weight: Optional[float] = None
    rotation_axis_weights: Optional[np.ndarray] = None
    yaw_target_deg: Optional[float] = None


class SimulationRequest(BaseModel):
    """Input accepted for one new website simulation.

    Attributes:
        seed: Optional Robosuite placement seed. ``None`` uses a new layout.
        target_classes: Objects to pick in their requested order. ``None``
            requests every class the detector knows about.
    """

    seed: int | None = None
    target_classes: list[Literal["Cereal", "Milk", "Can", "Bread"]] | None = Field(
        default=None, min_length=1
    )


class SimulationJob(BaseModel):
    """Public status of one queued or completed browser simulation."""

    id: str
    status: Literal["queued", "running", "paused", "completed", "failed", "stopped"]
    seed: int | None
    target_classes: list[str]
    placed_objects: int | None = None
    avg_detection_confidence: float | None = None
    avg_pose_position_error_cm: float | None = None
    detection_confidences: list[float] | None = None
    pose_position_errors_cm: list[float] | None = None
    error: str | None = None


class LiveProgress(BaseModel):
    """Current size of one run's in-progress trajectory stream.

    Attributes:
        ready: Whether the worker has written ``live.meta.json`` yet. This
            file appears within seconds of a run starting, well before the
            first frame. This is ``False`` only during that brief startup
            window.
        status: The owning simulation job's latest status.
        nq: Number of generalized-position values in every frame.
        frame_count: Frames currently available from ``/live/frames``.
        object_names: Object classes in this run's placement order.
        stage_names: Every named pick-and-place stage, in enum order.
    """

    ready: bool
    status: Literal["queued", "running", "paused", "completed", "failed", "stopped"]
    nq: int | None = None
    frame_count: int = 0
    object_names: list[str] = Field(default_factory=list)
    stage_names: list[str] = Field(default_factory=list)
