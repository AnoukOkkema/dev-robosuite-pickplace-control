from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    # Only needed for type hints, so don't import it at runtime.
    from src.control.stage import Stage


@dataclass
class DetectorConfig:
    """Settings for the ONNX object detector."""

    model_path: str = "models/yolo_detector.onnx"
    class_names: List[str] = field(
        default_factory=lambda: ["Bread", "Can", "Cereal", "Milk"]
    )
    image_size: int = 640
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45


@dataclass
class PoseConfig:
    """Settings for the ONNX pose estimator."""

    model_path: str = "models/pose_estimator.onnx"
    pos_image_size: int = 224
    rotation_image_size: int = 128
    rotation_symmetric_classes: List[str] = field(default_factory=list)


@dataclass
class EnvironmentConfig:
    """Simulator settings needed to reproduce a placement scene."""

    seed: Optional[int] = 42
    robot_base_offset: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])


@dataclass
class ExecutionConfig:
    """Object order for one pick-and-place run."""

    target_classes: List[str] = field(
        default_factory=lambda: ["Cereal", "Milk", "Can", "Bread"]
    )


@dataclass
class VisualizationConfig:
    """Optional image and MP4 output settings."""

    video_enabled: bool = False
    video_path: str = "outputs/videos/pickplace.mp4"
    video_fps: int = 10
    video_capture_every_ticks: int = 6
    trajectory_path: str = "docs/assets/sample-trajectory.bin"
    trajectory_model_path: str = "docs/assets/scene-model.zip"
    scan_images_enabled: bool = False
    scan_images_dir: str = "outputs/scans"


@dataclass
class MotionConfig:
    """Gains and per-tick limits for joint-space motion."""

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
    """Heights, tolerances, and dwell durations for each stage."""

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
    """Top-level application configuration loaded from YAML."""

    detector: DetectorConfig = field(default_factory=DetectorConfig)
    pose: PoseConfig = field(default_factory=PoseConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    stages: StagesConfig = field(default_factory=StagesConfig)


@dataclass
class Detection:
    """One bounding-box detection in image coordinates."""

    box: Tuple[float, float, float, float]
    confidence: float
    class_id: int


@dataclass
class CropRegion:
    """Crop coordinates in image pixels; the start is included, the end is not."""

    y1: int
    y2: int
    x1: int
    x2: int


@dataclass
class ImageSize:
    """Image dimensions in pixels."""

    height: int
    width: int


@dataclass
class PlacementTarget:
    """World-frame object pose and target-bin position."""

    class_name: str
    object_pos: np.ndarray  # World-frame XYZ position with shape (3,).
    yaw_deg: float  # World-frame yaw in degrees.
    bin_pos: np.ndarray  # World-frame target-bin XYZ position with shape (3,).


@dataclass
class RunResult:
    """Outcome summary for one completed pick-and-place run."""

    placed: int
    avg_detection_confidence: Optional[float] = None
    avg_pose_position_error_cm: Optional[float] = None
    avg_pose_rotation_error_deg: Optional[float] = None
    detection_confidences: List[float] = field(default_factory=list)
    pose_position_errors_cm: List[float] = field(default_factory=list)
    pose_rotation_errors_deg: List[Optional[float]] = field(default_factory=list)


@dataclass
class ControllerRunState:
    """State that persists while one object moves through the stage sequence."""

    stage: Stage
    stage_tick: int = 0
    grasp_joint7_target: Optional[float] = None
    rezero_orientation: Optional[np.ndarray] = None
    rezero_yaw_deg: Optional[float] = None


@dataclass(frozen=True)
class WristTargets:
    """Desired wrist joints for a vertical-down gripper."""

    joint5: Optional[float]
    joint6: float


@dataclass(frozen=True)
class PoseCommand:
    """One Cartesian command for the absolute joint-position controller."""

    position: np.ndarray
    orientation: np.ndarray
    gripper_action: float
    position_tolerance: float
    max_position_delta: Optional[float] = None
    joint7_target: Optional[float] = None
    orientation_weight: Optional[float] = None
    rotation_axis_weights: Optional[np.ndarray] = None
    yaw_target_deg: Optional[float] = None
