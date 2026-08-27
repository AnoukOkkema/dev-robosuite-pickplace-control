"""Load human-editable YAML settings into typed Python configuration objects.

Keeping parsing here separates configuration-file spelling and validation from
the vision, simulation, and robot-control code that consumes the settings.
"""

from pathlib import Path
from typing import Any, Mapping

import yaml

from src.util.types import (
    Config,
    DetectorConfig,
    EnvironmentConfig,
    ExecutionConfig,
    MotionConfig,
    PoseConfig,
    StagesConfig,
    VisualizationConfig,
)

# Keep this relative path so commands run from the repository root are portable.
CONFIG_PATH = Path("config", "config.yaml")


class ConfigReader:
    """Read and parse YAML configuration from disk.

    Attributes:
        _config_path: Location of the YAML file to read.
    """

    def __init__(self, config_path: Path = CONFIG_PATH):
        """Store the YAML file path.

        Args:
            config_path: Location of the application configuration file.
        """
        self._config_path = config_path

    def read(self) -> dict[str, Any]:
        """Read the YAML file and return its root mapping.

        Raises:
            FileNotFoundError: If the configuration file does not exist.
            RuntimeError: If the file cannot be read.
            TypeError: If the YAML root is not a mapping.
            ValueError: If the YAML syntax is invalid.
        """
        if not self._config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found at '{self._config_path.resolve()}'"
            )

        try:
            with open(self._config_path, "r") as file:
                raw = yaml.safe_load(file)
        except yaml.YAMLError as error:
            raise ValueError(
                f"Invalid YAML syntax in '{self._config_path}': {error}"
            ) from error
        except OSError as error:
            raise RuntimeError(
                f"Failed to read config file '{self._config_path}': {error}"
            ) from error

        if not isinstance(raw, dict):
            raise TypeError(
                "Invalid data in config structure: expected mapping at root, "
                f"got {type(raw).__name__}"
            )
        return raw


class ConfigAssembler:
    """Assemble typed configuration objects from a raw YAML mapping.

    The application reads uppercase YAML keys, while the returned dataclasses
    expose lowercase, type-checked Python attributes.
    """

    def assemble(self, raw: Mapping[str, Any]) -> Config:
        """Build typed configuration objects from a raw YAML mapping.

        Args:
            raw: Parsed YAML configuration.

        Returns:
            Validated configuration object used by the pipeline.

        Raises:
            KeyError: If a required YAML key is missing.
            TypeError: If a YAML value has an incompatible type.
        """
        try:
            return Config(
                detector=DetectorConfig(
                    model_path=raw["DETECTOR"]["MODEL_PATH"],
                    class_names=raw["DETECTOR"]["CLASS_NAMES"],
                    image_size=raw["DETECTOR"]["IMAGE_SIZE"],
                    conf_threshold=raw["DETECTOR"]["CONF_THRESHOLD"],
                    iou_threshold=raw["DETECTOR"]["IOU_THRESHOLD"],
                ),
                pose=PoseConfig(
                    model_path=raw["POSE"]["MODEL_PATH"],
                    pos_image_size=raw["POSE"]["POS_IMAGE_SIZE"],
                    rotation_image_size=raw["POSE"]["ROTATION_IMAGE_SIZE"],
                ),
                environment=EnvironmentConfig(
                    seed=raw["ENVIRONMENT"]["SEED"],
                    robot_base_offset=raw["ENVIRONMENT"]["ROBOT_BASE_OFFSET"],
                ),
                execution=ExecutionConfig(
                    target_classes=raw["EXECUTION"]["TARGET_CLASSES"],
                ),
                visualization=VisualizationConfig(
                    video_enabled=raw["VISUALIZATION"]["VIDEO_ENABLED"],
                    video_path=raw["VISUALIZATION"]["VIDEO_PATH"],
                    video_fps=raw["VISUALIZATION"]["VIDEO_FPS"],
                    video_capture_every_ticks=raw["VISUALIZATION"].get(
                        "VIDEO_CAPTURE_EVERY_TICKS", 6
                    ),
                    trajectory_enabled=raw["VISUALIZATION"].get(
                        "TRAJECTORY_ENABLED", False
                    ),
                    trajectory_path=raw["VISUALIZATION"].get(
                        "TRAJECTORY_PATH", "outputs/trajectory/trajectory.bin"
                    ),
                    trajectory_model_path=raw["VISUALIZATION"].get(
                        "TRAJECTORY_MODEL_PATH", "outputs/trajectory/model.zip"
                    ),
                    scan_images_enabled=raw["VISUALIZATION"].get(
                        "SCAN_IMAGES_ENABLED", False
                    ),
                    scan_images_dir=raw["VISUALIZATION"].get(
                        "SCAN_IMAGES_DIR", "outputs/scans"
                    ),
                ),
                motion=MotionConfig(
                    joint5_name=raw["MOTION"].get("JOINT5_NAME"),
                    joint5_kp=raw["MOTION"]["JOINT5_KP"],
                    joint5_down_tolerance_deg=raw["MOTION"][
                        "JOINT5_DOWN_TOLERANCE_DEG"
                    ],
                    joint6_name=raw["MOTION"]["JOINT6_NAME"],
                    lock_joint6=raw["MOTION"]["LOCK_JOINT6"],
                    joint6_kp=raw["MOTION"]["JOINT6_KP"],
                    joint6_down_tolerance_deg=raw["MOTION"][
                        "JOINT6_DOWN_TOLERANCE_DEG"
                    ],
                    joint7_kp=raw["MOTION"]["JOINT7_KP"],
                    yaw_joint_index=raw["MOTION"]["YAW_JOINT_INDEX"],
                    yaw_max_step_deg=raw["MOTION"]["YAW_MAX_STEP_DEG"],
                    max_joint_delta=raw["MOTION"]["MAX_JOINT_DELTA"],
                    max_wrist_joint_delta=raw["MOTION"]["MAX_WRIST_JOINT_DELTA"],
                    position_kp=raw["MOTION"]["POSITION_KP"],
                    orientation_kp=raw["MOTION"]["ORIENTATION_KP"],
                    orientation_weight=raw["MOTION"]["ORIENTATION_WEIGHT"],
                    max_position_delta=raw["MOTION"]["MAX_POSITION_DELTA"],
                    max_orientation_delta=raw["MOTION"]["MAX_ORIENTATION_DELTA"],
                    ik_damping=raw["MOTION"]["IK_DAMPING"],
                    gripper_clamp_force_multiplier=raw["MOTION"][
                        "GRIPPER_CLAMP_FORCE_MULTIPLIER"
                    ],
                    joint6_home_deg=raw["MOTION"].get("JOINT6_HOME_DEG"),
                ),
                stages=StagesConfig(
                    approach_height=raw["STAGES"]["APPROACH_HEIGHT"],
                    descend_clearance=raw["STAGES"]["DESCEND_CLEARANCE"],
                    fine_descend_offsets=raw["STAGES"]["FINE_DESCEND_OFFSETS"],
                    fine_descend_max_position_deltas=raw["STAGES"].get(
                        "FINE_DESCEND_MAX_POSITION_DELTAS", {}
                    ),
                    bin_release_heights=raw["STAGES"]["BIN_RELEASE_HEIGHTS"],
                    position_tolerance=raw["STAGES"]["POSITION_TOLERANCE"],
                    bin_release_tolerance=raw["STAGES"]["BIN_RELEASE_TOLERANCE"],
                    yaw_tolerance_deg=raw["STAGES"]["YAW_TOLERANCE_DEG"],
                    grasp_dwell_ticks=raw["STAGES"]["GRASP_DWELL_TICKS"],
                    grasp_min_lift_height=raw["STAGES"]["GRASP_MIN_LIFT_HEIGHT"],
                    grasp_yaw_offset_deg=raw["STAGES"]["GRASP_YAW_OFFSET_DEG"],
                    release_dwell_ticks=raw["STAGES"]["RELEASE_DWELL_TICKS"],
                    max_ticks_per_stage=raw["STAGES"]["MAX_TICKS_PER_STAGE"],
                    log_every_ticks=raw["STAGES"]["LOG_EVERY_TICKS"],
                    open_gripper=raw["STAGES"]["OPEN_GRIPPER"],
                    closed_gripper=raw["STAGES"]["CLOSED_GRIPPER"],
                ),
            )
        except KeyError as error:
            raise KeyError(f"Missing required key in config file: {error}") from error
        except TypeError as error:
            raise TypeError(f"Invalid data in config structure: {error}") from error



class SystemConfigurator:
    """Loads and assembles the app's typed SystemConfig from config.yaml."""

    DEFAULT_CONFIG_PATH = Path("config", "config.yaml")

    @classmethod
    def load(cls, config_path: Path = DEFAULT_CONFIG_PATH) -> Config:
        """
        Reads and assembles the typed SystemConfig.

        Args:
            config_path (Path): Path to the YAML config file.

        Returns:
            SystemConfig: The assembled, typed configuration.
        """

        raw = ConfigReader(config_path).read()
        return ConfigAssembler().assemble(raw)
