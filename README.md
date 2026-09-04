# dev-robosuite-pickplace-control

A robot control pipeline for a pick-and-place task. It drives a robosuite `PickPlace` simulation (Panda arm + Robotiq85 gripper), using the ONNX models trained by [`dev-robosuite-pickplace-vision`](https://github.com/AnoukOkkema/dev-robosuite-pickplace-vision).

`yolo_detector.onnx` finds each object's box and class. `pose_estimator.onnx` predicts its position and rotation, in camera-frame xyz. An 11-stage joint-space state machine then moves the arm to pick each object up and place it in its bin.

<video src="https://raw.githubusercontent.com/AnoukOkkema/dev-robosuite-pickplace-control/main/outputs/videos/pickplace_frontview.mp4" controls muted title="Front-view recording of a pick-and-place run"></video>

## Table of contents

- [Highlights](#highlights)
- [Requirements](#requirements)
- [Installation](#installation)
- [Project structure](#project-structure)
- [Quick start](#quick-start)
- [Configuration](#configuration-configconfigyaml)
- [Pipeline](#pipeline-mainpy)
- [Recording](#recording-baserunobserver)
- [Code quality](#code-quality)
- [Citation](#citation)

## Highlights

- Full joint-space control: the code works out each of the arm's joint angles directly, every tick, instead of leaving that to a black-box controller. Moving the arm and keeping the gripper pointed straight down are handled by two separate calculations that never touch the same joints, so they can't undo each other's work.
- An 11-stage state machine per object: raise, align, descend, orient yaw, fine descend, grasp, lift, re-zero yaw, move to bin, descend into bin, release. Each of the four objects (visually and geometrically different) gets its own tuning: grasp offsets, release heights, and speed caps.
- A fresh detection scan runs before every object, not just once at the start. So an object that started out hidden behind another one becomes pickable once that object is gone.
- Recording is fully separate from the control logic, using an observer pattern (`BaseRunObserver`). MP4 export, a compact trajectory format for the hosted [`docs/`](docs/) demo player, and per-scan diagnostic screenshots are three independent listeners on the same five lifecycle hooks. Video and scan images can each be turned on or off; the trajectory export always runs. The executor doesn't know any of them exist.

## Requirements

- Python >= 3.11, < 3.12
- [uv](https://docs.astral.sh/uv/) for dependency management
- `models/yolo_detector.onnx` and `models/pose_estimator.onnx` (trained and exported by [`dev-robosuite-pickplace-vision`](https://github.com/AnoukOkkema/dev-robosuite-pickplace-vision); copy the latest exported models over if you're updating them)
- macOS, Windows, or Linux. An NVIDIA GPU is optional (see [Installation](#installation))

## Installation

This project uses [uv](https://docs.astral.sh/uv/). Pick the extra that matches your hardware:

```bash
uv sync --extra cpu
```

```bash
uv sync --extra mps
```

```bash
uv sync --extra gpu
```

`cpu` and `mps` use the standard torch wheels (CPU / macOS with MPS support). `gpu` installs torch with CUDA 12.4 and `onnxruntime-gpu`.

Detection and pose inference run on ONNX Runtime. They use an NVIDIA GPU automatically when one is available (`CUDAExecutionProvider`), and fall back to CPU otherwise (`src/vision/onnx_detector.py`, `src/vision/pose_estimator.py`).

## Project structure

```
dev-robosuite-pickplace-control/
├── config/
│   ├── config.yaml                     # detector/pose/environment/motion/stage tuning (see below)
│   ├── joint_position_controller.json  # robosuite composite controller config (see below)
│   └── logging.yaml                    # stdlib logging.dictConfig
├── docs/                    # hosted GitHub Pages demo player (see docs/README.md)
│   ├── index.html
│   └── assets/
│       ├── script.js                      # viewer bundle
│       ├── style.css                      # viewer styles
│       ├── mujoco.wasm                    # MuJoCo physics engine, compiled to WebAssembly
│       ├── sample-trajectory.bin          # one recorded run's qpos frames
│       ├── sample-trajectory.result.json  # that run's placement/vision-accuracy summary
│       └── scene-model.zip                # packaged MJCF scene model
├── models/
│   ├── yolo_detector.onnx   # copied from dev-robosuite-pickplace-vision
│   └── pose_estimator.onnx  # copied from dev-robosuite-pickplace-vision
├── main.py                  # entrypoint
└── src/
    ├── control/
    │   ├── joint_motion.py              # differential IK + J5/J6 posture control
    │   ├── pickplace_controller.py      # 11-stage state machine
    │   └── stage.py                     # shared stage order
    ├── environments/
    │   └── pickplace_with_robot_offset.py # PickPlace env + robot base offset
    ├── execution/
    │   └── pickplace_executor.py        # builds env, runs vision, drives PickPlaceController
    ├── geometry/
    │   └── coordinate_transform.py      # camera-frame -> world-frame, rotation -> yaw
    ├── recording/
    │   ├── run_observer.py              # BaseRunObserver hook contract
    │   ├── factory.py                   # builds the observers one run's config enables
    │   ├── trajectory_recorder.py       # portable model + qpos recording for browser playback
    │   ├── trajectory_observer.py       # drives TrajectoryRecorder from executor hooks
    │   ├── video_recorder.py            # agentview + frontview MP4 writer
    │   ├── video_observer.py            # drives VideoRecorder, adds detection/pose overlays
    │   └── scan_image_observer.py       # saves an annotated PNG per detection scan
    ├── util/
    │   ├── logging_configurator.py      # logging.yaml-driven setup
    │   ├── system_configurator.py       # ConfigReader/ConfigAssembler/SystemConfigurator.load()
    │   └── types.py                     # shared configuration and state data
    └── vision/
        ├── onnx_detector.py             # YOLO ONNX inference
        └── pose_estimator.py            # pose_estimator.onnx inference
```

## Quick start

1. Install dependencies (see above) and make sure `models/yolo_detector.onnx` and `models/pose_estimator.onnx` are present.
2. Check `config/config.yaml`. `VISUALIZATION.VIDEO_ENABLED` decides whether the run opens an interactive viewer (`false`, the default) or runs headless and records an MP4 (`true`), see [Configuration](#configuration-configconfigyaml). Every run exports a trajectory either way. Keep in mind that MP4 recording takes around 7.5 minutes, since it renders and overlays two extra Full-HD frames every few ticks.
3. Run the pipeline:

   ```bash
   # macOS
   uv run mjpython main.py

   # Windows / Linux
   uv run python main.py
   ```

   macOS needs `mjpython` for the interactive viewer, since MuJoCo's viewer has to run on the main thread; headless runs (recording video) work fine with plain `python` there too. Windows and Linux always use plain `python`.

## Configuration (`config/config.yaml`)

**`DETECTOR`**

| Key | Description |
|---|---|
| `MODEL_PATH` | ONNX detector loaded for inference. |
| `CLASS_NAMES` | Class names indexed by the model's class ID (fixed training order). |
| `IMAGE_SIZE` | Square input resolution (pixels). |
| `CONF_THRESHOLD` | Minimum detection confidence to keep a box. |
| `IOU_THRESHOLD` | IoU threshold used by NMS to suppress overlapping boxes. |

**`POSE`**

| Key | Description |
|---|---|
| `MODEL_PATH` | ONNX `PoseEstimator` model. |
| `POS_IMAGE_SIZE` | Input size of the full agentview frame fed to the xyz stream. |
| `ROTATION_IMAGE_SIZE` | Input size of the cropped object image fed to the rotation stream. |
| `ROTATION_SYMMETRIC_CLASSES` | Classes with no meaningful "correct" rotation to compare against (e.g. a can looks and grasps the same at any yaw). Excluded from the per-scan rotation-error logging and the `RunResult` average, both of which compare the vision model's predictions against the simulator's own ground-truth object poses (`PickPlaceExecutor._pose_errors`). Position is unaffected. |

**`ENVIRONMENT`**

| Key | Description |
|---|---|
| `SEED` | robosuite scene-layout seed; `null` randomizes the layout every run. |
| `ROBOT_BASE_OFFSET` | World-frame `[x, y, z]` offset added on top of robosuite's own default Panda base position. |

**`EXECUTION`**

| Key | Description |
|---|---|
| `TARGET_CLASSES` | Classes and the order the run picks and places them in. Independent of `DETECTOR.CLASS_NAMES`'s alphabetical class-ID order. |

**`VISUALIZATION`**

| Key | Description |
|---|---|
| `VIDEO_ENABLED` | Record agent-view/front-view MP4s. |
| `VIDEO_PATH` | Base output path for the generated MP4s. |
| `VIDEO_FPS` | Output video playback frame rate. |
| `VIDEO_CAPTURE_EVERY_TICKS` | Controller ticks between recorded frames. |
| `TRAJECTORY_PATH` | Output binary path for the recorded trajectory (exported on every run, consumed by the hosted [`docs/`](docs/) demo player). |
| `TRAJECTORY_MODEL_PATH` | Output ZIP path for the packaged MJCF scene model. |
| `SCAN_IMAGES_ENABLED` | Save an annotated PNG per detection scan, plus one raw opening capture. |
| `SCAN_IMAGES_DIR` | Directory the scan PNGs are written into. |

**`MOTION`**

| Key | Description |
|---|---|
| `JOINT5_NAME` | robosuite joint name for wrist joint J5; `null` disables the joint J5+J6 solve in favor of a J6-only fallback. |
| `JOINT5_KP` | Proportional gain for the J5 tracking correction. |
| `JOINT5_DOWN_TOLERANCE_DEG` | J5 convergence tolerance for "pointing straight down". |
| `JOINT6_NAME` | robosuite joint name for wrist joint J6. |
| `LOCK_JOINT6` | If true, hard-overwrites the final J6 target after IK instead of trusting convergence. |
| `JOINT6_KP` | Proportional gain for the J6 tracking correction. |
| `JOINT6_DOWN_TOLERANCE_DEG` | J6 convergence tolerance. |
| `JOINT7_KP` | Proportional gain for the explicit Joint-7 branch of `JointMotion.action()`. |
| `YAW_JOINT_INDEX` | Arm-action array index of the yaw joint (J7). |
| `YAW_MAX_STEP_DEG` | Per-tick cap on yaw-stage correction. |
| `MAX_JOINT_DELTA` | Per-tick cap on the IK solution's largest joint-angle component. |
| `MAX_WRIST_JOINT_DELTA` | Separate, larger per-tick cap for the J5/J6 tracking correction. |
| `POSITION_KP` | Proportional gain on Cartesian position error. |
| `ORIENTATION_KP` | Proportional gain on orientation error. |
| `ORIENTATION_WEIGHT` | Base weight of orientation correction; `0.0` unless a stage explicitly holds world yaw. |
| `MAX_POSITION_DELTA` | Default per-tick Cartesian step cap (metres); stages may override it. |
| `MAX_ORIENTATION_DELTA` | Per-tick cap (radians) on orientation correction. |
| `IK_DAMPING` | Damping added to the IK solve to keep it stable, especially near tricky arm poses. |
| `GRIPPER_CLAMP_FORCE_MULTIPLIER` | Multiplier applied to the Robotiq85 finger actuator force. |
| `JOINT6_HOME_DEG` *(optional, unset)* | If set, each new object's J6 target starts here instead of the robot's current J6 angle. |

**`STAGES`**

| Key | Description |
|---|---|
| `APPROACH_HEIGHT` | Shared world-Z used for safe horizontal travel between stages. |
| `DESCEND_CLEARANCE` | Z clearance above the object for the coarse descend stage. |
| `FINE_DESCEND_OFFSETS` | Per-class final-approach Z offset, correcting each mesh's real contact height vs. its reported origin. |
| `FINE_DESCEND_MAX_POSITION_DELTAS` | Per-class cap on per-tick position change during fine descent, gentler than `MAX_POSITION_DELTA`. |
| `BIN_RELEASE_HEIGHTS` | Per-class absolute end-effector world-Z at release. |
| `POSITION_TOLERANCE` | Default Cartesian convergence tolerance. |
| `BIN_RELEASE_TOLERANCE` | Tighter tolerance specific to the bin-release stage. |
| `YAW_TOLERANCE_DEG` | Yaw convergence tolerance. |
| `GRASP_DWELL_TICKS` | Ticks the gripper is held closing before checking the grasp. |
| `GRASP_MIN_LIFT_HEIGHT` | Minimum object rise (metres) required before a lift counts as successful. |
| `GRASP_YAW_OFFSET_DEG` | Constant added to the detected grasp yaw before symmetry-folding. |
| `RELEASE_DWELL_TICKS` | Ticks the gripper is held opening on release. |
| `MAX_TICKS_PER_STAGE` | Per-stage tick budget; exceeding it raises instead of correcting forever. |
| `LOG_EVERY_TICKS` | Interval for periodic progress log lines. |
| `OPEN_GRIPPER` / `CLOSED_GRIPPER` | Gripper actuator command values. |

## Pipeline (`main.py`)

Five steps, run for every object in every simulated episode:

1. Environment: build the robosuite `PickPlace` simulation, with a configurable robot base offset.
2. Detection: find each visible object's box and class in the agent-view frame.
3. Pose estimation: predict each detected object's position and rotation.
4. Coordinate transform: convert predicted camera-frame poses into world-frame poses the robot can act on.
5. Control: drive the arm through an 11-stage state machine to pick up the object and place it in its bin.

### Step 1: Environment (`PickPlaceWithRobotOffset`)

A minimal subclass of robosuite's `PickPlace` task, making the Panda's base position configurable via `ENVIRONMENT.ROBOT_BASE_OFFSET`.

`PickPlaceExecutor._init_env` builds it with a Panda arm, a Robotiq85 gripper, and `config/joint_position_controller.json` as the controller config, putting the pipeline in joint-space control instead of robosuite's default operational-space control. 

### Step 2: Detection (`OnnxDetector`)

`PickPlaceExecutor._detect_targets` crops the agentview frame to `CropRegion` and runs it through `OnnxDetector.predict()`:

- `_letterbox` resizes it into a `DETECTOR.IMAGE_SIZE` square, preserving aspect ratio with black padding.
- `_to_blob` converts BGR to RGB and normalizes it to NCHW.
- The ONNX session (`CUDAExecutionProvider`, falling back to `CPUExecutionProvider`) returns raw YOLO predictions.
- `_postprocess` filters those by `DETECTOR.CONF_THRESHOLD`, runs `cv2.dnn.NMSBoxes` at `DETECTOR.IOU_THRESHOLD`, and un-letterboxes the result back into the cropped image's pixel space.

### Step 3: Pose estimation (`PoseEstimator`)

For each kept detection, `PoseEstimator.predict()` builds four inputs, in the model's fixed order: the full agentview frame (`POSE.POS_IMAGE_SIZE`), normalized bbox coordinates plus derived area and center, a one-hot class vector, and the detection's own crop (`POSE.ROTATION_IMAGE_SIZE`).

The ONNX session returns a camera-frame `xyz` and a 6D rotation vector. `PoseEstimator.rot6d_to_matrix` turns that into a proper 3x3 rotation matrix via Gram-Schmidt orthogonalization (the Zhou et al., 2019 representation, matching the training convention in [`dev-robosuite-pickplace-vision`](https://github.com/AnoukOkkema/dev-robosuite-pickplace-vision)).

### Step 4: Coordinate transform (`CoordinateTransformer`)

The vision model predicts each object's position and rotation relative to the camera, but the arm plans its motion in world coordinates. `CoordinateTransformer` converts one into the other, using the camera's
own known position and orientation in the scene.

It also reduces the full 3D rotation down to a single grasp yaw angle, folded into `[-90°, 90°]`, since turning any of the four objects 180° around the vertical axis gives an equally valid grasp.

### Step 5: Control (`PickPlaceController`, `JointMotion`)

`PickPlaceController.run()` drives one object through an 11-stage state machine (`Stage`, [`src/control/stage.py`](src/control/stage.py)):

1. **RAISE**: rise to a shared safe travel height above the table.
2. **ALIGN_XY**: move horizontally to the detected object, still at travel height.
3. **DESCEND**: drop to just above the object. `Can` skips straight to `FINE_DESCEND`; every other class goes through `ORIENT_YAW` first.
4. **ORIENT_YAW**: rotate joint 7 only, toward the detected grasp yaw.
5. **FINE_DESCEND**: the final, slower approach to contact height, with a per-class Z correction.
6. **GRASP**: hold pose, close the gripper, and verify contact.
7. **LIFT**: rise back to travel height. Requires the object to still be grasped and to have risen a minimum height.
8. **REZERO_YAW**: a second joint-7-only stage, resetting the wrist to a fixed orientation before transport.
9. **MOVE_TO_BIN**: travel to the target bin at travel height.
10. **DESCEND_TO_BIN**: drop to a per-class release height inside the bin.
11. **RELEASE**: hold pose and open the gripper.

Every stage carries a wrist target and only advances once both position and wrist orientation have converged. If a stage still hasn't converged after `STAGES.MAX_TICKS_PER_STAGE` ticks, the run stops with an error instead of retrying indefinitely.

`JointMotion.action()` ([`src/control/joint_motion.py`](src/control/joint_motion.py)) turns one target pose into a 7-joint action, every tick, always as one small step from the robot's current joints:

- **Position**: scaled and capped (`MOTION.MAX_POSITION_DELTA`).
- **Orientation**: off by default, only active for stages holding a world-frame yaw.
- **Position and yaw** come from a damped-least-squares differential IK solve, using the arm's Jacobian. Roll and pitch are left out, so they don't fight the separate J5/J6 wrist solve.
- **The J5/J6 wrist solve** works out, each tick, which angles keep the tool pointing straight down, tracked with a feedforward-plus-proportional correction.

Yaw-only stages (`ORIENT_YAW`/`REZERO_YAW`) hold every other joint fixed and step only joint 7, so a yaw correction never moves the gripper off the object.

## Recording (`BaseRunObserver`)

Recording is entirely separate from the pick-and-place logic. `PickPlaceExecutor` only reports what happens, through five lifecycle hooks defined by the `BaseRunObserver` ABC ([`src/recording/run_observer.py`](src/recording/run_observer.py)):

- `on_run_start`: once, after `env.reset()`.
- `on_scan` / `on_target_detected`: per detection scan, before every object.
- `on_step`: after every controller tick.
- `on_run_end`: once, in a `finally` block, with the run's `RunResult` summary, so a crashed or early-terminated run still gets whatever it captured.

[`src/recording/factory.py`](src/recording/factory.py)'s `build_observers()` reads `VisualizationConfig` and always builds the trajectory observer, plus whichever of video and scan images are turned on. 

### Video (`VideoObserver`)

The observer writes two Full-HD (1920x1080) MP4s when `VISUALIZATION.VIDEO_ENABLED` is `true` and the headless command above is used: `outputs/videos/pickplace_agentview.mp4` and `outputs/videos/pickplace_frontview.mp4`. They're downsampled to one captured tick every `VIDEO_CAPTURE_EVERY_TICKS` (6 by default), to bound render time, using a fast H.264 preset. 

The detection overlay (box, label, red-X/green-Y/blue-Z pose axes) is only drawn during the pre-grasp approach stages (`RAISE` through `FINE_DESCEND`). It marks where the scan detected the object, which stops being accurate the moment the gripper starts moving it.

### Trajectory (`TrajectoryObserver`)

The observer writes a compact recording on every run, consumed by the hosted [`docs/`](docs/) demo player. It writes three files, derived from `TRAJECTORY_PATH`/`TRAJECTORY_MODEL_PATH`:

- `model.zip`: the run's packaged MJCF scene, exported at `on_run_start`.
- `trajectory.bin`: the final recording, a compact length-prefixed-header binary, written once at `on_run_end`.
- `<trajectory-stem>.result.json`: the run's placement/vision-accuracy summary, written alongside `trajectory.bin` at `on_run_end`, so a run produces everything the demo player needs in one place.

### Scan images (`ScanImageObserver`)

The observer saves an annotated PNG per detection scan, on by default (`VISUALIZATION.SCAN_IMAGES_ENABLED`): `outputs/scans/scan_NN_agentview.png` (box, label, and pose axes for every detected object, not just the one currently sought). The very first scan also saves an unannotated `outputs/scans/raw_agentview.png` of the opening scene, for documentation. Set `SCAN_IMAGES_DIR` to change where they're written.

## Code quality

Install the Git hook once after cloning:

```bash
uv run pre-commit install
```

Run every hook manually when needed:

```bash
uv run pre-commit run --all-files
```

## Citation

This project uses [robosuite](https://robosuite.ai/). If you use robosuite in
work based on this project, cite:

```bibtex
@inproceedings{robosuite2020,
  title={robosuite: A Modular Simulation Framework and Benchmark for Robot Learning},
  author={Yuke Zhu and Josiah Wong and Ajay Mandlekar and Roberto Mart\'{i}n-Mart\'{i}n and Abhishek Joshi and Soroush Nasiriany and Yifeng Zhu and Kevin Lin},
  booktitle={arXiv preprint arXiv:2009.12293},
  year={2020}
}
```
