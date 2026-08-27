# dev-robot-pickplace-control

Drives a robosuite `PickPlace` simulation (Panda arm + Robotiq85 gripper)
through a full pick-and-place cycle using the ONNX models produced by
[`dev-robosuite-pickplace-vision`](../dev-robosuite-pickplace-vision):
`yolo_detector.onnx` detects each object's bbox/class, `pose_estimator.onnx`
predicts its camera-frame xyz + rotation, and an 11-stage joint-space state
machine drives the arm to pick each object up and place it in its bin.

![Detection box and pose axes on a scanned object, agent-view camera](outputs/scans/scan_01_agentview.png)

## Table of contents

- [Highlights](#highlights)
- [Requirements](#requirements)
- [Installation](#installation)
- [Running](#running)
- [Configuration](#configuration-configconfigyaml)
- [Pipeline](#pipeline-mainpy)
- [Recording](#recording-baserunobserver)
- [Local simulation website](#local-simulation-website)
- [Project structure](#project-structure)
- [Code quality](#code-quality)
- [Citation](#citation)

## Highlights

- Fully joint-space control: a damped-least-squares differential IK solve
  (position + yaw) runs every tick alongside an independent local-optimization
  solve for the J5/J6 wrist joints, continuously re-aimed to keep the gripper
  pointing straight down. The two never fight over the same joints, because
  the IK task Jacobian excludes J5/J6 entirely.
- An 11-stage state machine per object (raise → align → descend → orient yaw
  → fine descend → grasp → lift → re-zero yaw → move to bin → descend into
  bin → release), with per-object-class tuning — grasp offsets, release
  heights, per-tick speed caps — for four visually and geometrically
  different objects.
- A fresh detection scan runs before *every* object, not just once at the
  start, so an object that started out occluded becomes pickable once the
  object in front of it is gone.
- Recording is fully decoupled from control via an observer pattern
  (`BaseRunObserver`): MP4 export, a compact binary trajectory for browser
  replay, and per-scan diagnostic screenshots are three independent,
  toggleable listeners on the same five lifecycle hooks — the executor has
  no idea any of them exist.
- A local web app replays exact recorded MuJoCo states in the browser
  (Three.js + MuJoCo-WASM) while a separate headless run streams live
  progress over a hand-rolled, byte-exact binary protocol — no server-side
  rendering, no video streaming.

## Requirements

- Python >= 3.11, < 3.12
- [uv](https://docs.astral.sh/uv/) for dependency management
- `models/yolo_detector.onnx` and `models/pose_estimator.onnx` (trained and
  exported by [`dev-robosuite-pickplace-vision`](../dev-robosuite-pickplace-vision))
- macOS, Windows, or Linux — an NVIDIA GPU is optional (see
  [Installation](#installation))
- Node.js + npm, only if you want the
  [local simulation website](#local-simulation-website)

## Installation

This project uses [uv](https://docs.astral.sh/uv/). Pick the extra that
matches your hardware:

```bash
uv sync --extra cpu
```

```bash
uv sync --extra mps
```

```bash
uv sync --extra gpu
```

`cpu` and `mps` use the standard torch wheels (CPU / macOS with MPS
support), `gpu` installs torch with CUDA 12.4 and `onnxruntime-gpu`.

Both running modes below require `models/yolo_detector.onnx` and
`models/pose_estimator.onnx` to be present (copy the latest trained/exported
models from `dev-robosuite-pickplace-vision` if updating).

Detection and pose inference run on ONNX Runtime and use an NVIDIA GPU via
`CUDAExecutionProvider` automatically when it's available, falling back to
CPU otherwise (`src/vision/onnx_detector.py`, `src/vision/pose_estimator.py`).
macOS has no CUDA support, so inference always runs on CPU there regardless.

## Running

For the interactive viewer:

```bash
# macOS
uv run mjpython main.py

# Windows / Linux
uv run python main.py
```

For a headless run (no viewer window), on any OS:

```bash
uv run python -m main
```

macOS requires MuJoCo's interactive viewer to run on the main thread, which
is what `mjpython` (bundled with the `mujoco` package) provides. That
conflicts with the window context headless MuJoCo rendering uses, so on
macOS specifically, don't use `mjpython` for MP4 recording or for the web
API's worker process — both always run headless via plain `python`. Windows
and Linux don't have this constraint: the same `python` interpreter handles
both the interactive viewer and headless runs.

## Configuration (`config/config.yaml`)

[`src/utils/system_configurator.py`](src/utils/system_configurator.py) reads
`config/config.yaml` in two stages: `ConfigReader.read()` parses the YAML
into a raw `dict`, then `ConfigAssembler.assemble()` builds a typed `Config`
(`src/utils/types.py`) from it, one dataclass per section. `load_config()` is
the sole entrypoint. Most keys are read as required (a missing one raises
`KeyError`); only a handful of `VISUALIZATION.*`, `MOTION.JOINT5_NAME`,
`MOTION.JOINT6_HOME_DEG`, and `STAGES.FINE_DESCEND_MAX_POSITION_DELTAS` are
optional with a fallback. Because of this, the values below — not the
dataclass defaults in `types.py` — are what actually runs.

### `DETECTOR`

| Key | Description | Default |
|---|---|---|
| `MODEL_PATH` | ONNX detector loaded for inference. | `models/yolo_detector.onnx` |
| `CLASS_NAMES` | Class names indexed by the model's class ID (fixed training order). | `[Bread, Can, Cereal, Milk]` |
| `IMAGE_SIZE` | Square input resolution (pixels). | `640` |
| `CONF_THRESHOLD` | Minimum detection confidence to keep a box. | `0.50` |
| `IOU_THRESHOLD` | IoU threshold used by NMS to suppress overlapping boxes. | `0.45` |

### `POSE`

| Key | Description | Default |
|---|---|---|
| `MODEL_PATH` | ONNX `PoseEstimator` model. | `models/pose_estimator.onnx` |
| `POS_IMAGE_SIZE` | Input size of the full agentview frame fed to the xyz stream. | `224` |
| `ROTATION_IMAGE_SIZE` | Input size of the cropped object image fed to the rotation stream. | `128` |

### `ENVIRONMENT`

| Key | Description | Default |
|---|---|---|
| `SEED` | robosuite scene-layout seed; `null` randomizes the layout every run. | `null` |
| `ROBOT_BASE_OFFSET` | World-frame `[x, y, z]` offset added on top of robosuite's own default Panda base position. | `[0.10, 0.10, 0.0]` |

### `EXECUTION`

| Key | Description | Default |
|---|---|---|
| `TARGET_CLASSES` | Classes and the order the run picks and places them in — independent of `DETECTOR.CLASS_NAMES`'s alphabetical class-ID order. | `[Cereal, Milk, Can, Bread]` |

### `VISUALIZATION`

| Key | Description | Default |
|---|---|---|
| `VIDEO_ENABLED` | Record agent-view/front-view MP4s. | `false` |
| `VIDEO_PATH` | Base output path for the generated MP4s. | `outputs/pickplace.mp4` |
| `VIDEO_FPS` | Output video playback frame rate. | `30` |
| `VIDEO_CAPTURE_EVERY_TICKS` | Controller ticks between recorded frames. | `6` |
| `TRAJECTORY_ENABLED` | Export a compact `qpos` trajectory + reusable MJCF model for the web player. | `false` |
| `TRAJECTORY_PATH` | Output binary path for the recorded trajectory. | `outputs/trajectory/trajectory.bin` |
| `TRAJECTORY_MODEL_PATH` | Output ZIP path for the packaged MJCF scene model. | `outputs/trajectory/model.zip` |
| `SCAN_IMAGES_ENABLED` | Save an annotated PNG per detection scan, plus one raw opening capture. | `true` |
| `SCAN_IMAGES_DIR` | Directory the scan PNGs are written into. | `outputs/scans` |

### `MOTION`

| Key | Description | Default |
|---|---|---|
| `JOINT5_NAME` | robosuite joint name for wrist joint J5; `null` disables the joint J5+J6 solve in favor of a J6-only fallback. | `robot0_joint5` |
| `JOINT5_KP` | Proportional gain for the J5 tracking correction. | `4.0` |
| `JOINT5_DOWN_TOLERANCE_DEG` | J5 convergence tolerance for "pointing straight down". | `1.0` |
| `JOINT6_NAME` | robosuite joint name for wrist joint J6. | `robot0_joint6` |
| `LOCK_JOINT6` | If true, hard-overwrites the final J6 target after IK instead of trusting convergence. | `false` |
| `JOINT6_KP` | Proportional gain for the J6 tracking correction. | `4.0` |
| `JOINT6_DOWN_TOLERANCE_DEG` | J6 convergence tolerance. | `1.0` |
| `JOINT7_KP` | Proportional gain for the explicit Joint-7 branch of `JointMotion.action()`. | `4.0` |
| `YAW_JOINT_INDEX` | Arm-action array index of the yaw joint (J7). | `6` |
| `YAW_MAX_STEP_DEG` | Per-tick cap on yaw-stage correction. | `4.0` |
| `MAX_JOINT_DELTA` | Per-tick cap on the IK solution's largest joint-angle component. | `0.05` |
| `MAX_WRIST_JOINT_DELTA` | Separate, larger per-tick cap for the J5/J6 tracking correction. | `0.10` |
| `POSITION_KP` | Proportional gain on Cartesian position error. | `4.0` |
| `ORIENTATION_KP` | Proportional gain on orientation error. | `1.0` |
| `ORIENTATION_WEIGHT` | Base weight of orientation correction; `0.0` unless a stage explicitly holds world yaw. | `0.0` |
| `MAX_POSITION_DELTA` | Default per-tick Cartesian step cap (metres); stages may override it. | `0.03` |
| `MAX_ORIENTATION_DELTA` | Per-tick cap (radians) on orientation correction. | `0.10` |
| `IK_DAMPING` | Levenberg damping coefficient in the damped-least-squares IK solve. | `0.03` |
| `GRIPPER_CLAMP_FORCE_MULTIPLIER` | Multiplier applied to the Robotiq85 finger actuator force. | `1.05` |
| `JOINT6_HOME_DEG` *(optional, unset)* | If set, each new object's J6 target starts here instead of the robot's current J6 angle. | *unset* |

### `STAGES`

| Key | Description | Default |
|---|---|---|
| `APPROACH_HEIGHT` | Shared world-Z used for safe horizontal travel between stages. | `1.2` |
| `DESCEND_CLEARANCE` | Z clearance above the object for the coarse descend stage. | `0.12` |
| `FINE_DESCEND_OFFSETS` | Per-class final-approach Z offset, correcting each mesh's real contact height vs. its reported origin. | `{Cereal: 0.018, Milk: 0.022, Can: -0.003, Bread: -0.003}` |
| `FINE_DESCEND_MAX_POSITION_DELTAS` | Per-class cap on per-tick position change during fine descent, gentler than `MAX_POSITION_DELTA`. | `{Bread: 0.005, Can: 0.015, Cereal: 0.015, Milk: 0.005}` |
| `BIN_RELEASE_HEIGHTS` | Per-class absolute end-effector world-Z at release. | `{Cereal: 0.95, Milk: 0.95, Can: 0.95, Bread: 0.85}` |
| `POSITION_TOLERANCE` | Default Cartesian convergence tolerance. | `0.005` |
| `BIN_RELEASE_TOLERANCE` | Tighter tolerance specific to the bin-release stage. | `0.005` |
| `YAW_TOLERANCE_DEG` | Yaw convergence tolerance. | `0.8` |
| `GRASP_DWELL_TICKS` | Ticks the gripper is held closing before checking the grasp. | `40` |
| `GRASP_MIN_LIFT_HEIGHT` | Minimum object rise (metres) required before a lift counts as successful. | `0.03` |
| `GRASP_YAW_OFFSET_DEG` | Constant added to the detected grasp yaw before symmetry-folding. | `0.0` |
| `RELEASE_DWELL_TICKS` | Ticks the gripper is held opening on release. | `50` |
| `MAX_TICKS_PER_STAGE` | Per-stage tick budget; exceeding it raises instead of correcting forever. | `1500` |
| `LOG_EVERY_TICKS` | Interval for periodic progress log lines. | `20` |
| `OPEN_GRIPPER` / `CLOSED_GRIPPER` | Gripper actuator command values. | `-0.05` / `1.0` |

## Pipeline (`main.py`)

### Step 1 — Environment (`PickPlaceWithRobotOffset`, [`src/environments/pickplace_with_robot_offset.py`](src/environments/pickplace_with_robot_offset.py))

A minimal subclass of robosuite's `PickPlace` task whose only job is making
the Panda's base position configurable, without touching object/bin/reward/
observation logic. `PickPlaceExecutor._init_env` builds it via
`robosuite.suite.make(..., robots="Panda", gripper_types="Robotiq85Gripper",
...)` with `has_offscreen_renderer=False, use_camera_obs=False` — robosuite's
own camera pipeline is disabled entirely, because with it enabled robosuite
would render a full agentview frame on every single `env.step()`, while the
executor only needs one frame per detection scan (it renders directly via
`mujoco.Renderer` instead). `_load_model()` reads robosuite's own default
Panda base position for the "bins" arena
(`robot_model.base_xpos_offset["bins"]`) and adds `ENVIRONMENT.ROBOT_BASE_OFFSET`
on top of it — additively, not relative to the world origin — so the offset
can be used to stress-test the controller's reach margins without editing
robosuite itself; it's skipped entirely when the offset is all-zero.

### Step 2 — Detection (`OnnxDetector`, [`src/vision/onnx_detector.py`](src/vision/onnx_detector.py))

`PickPlaceExecutor._detect_targets` crops the agentview frame to
`CropRegion` and runs it through `OnnxDetector.predict()`: `_letterbox`
resizes into a `DETECTOR.IMAGE_SIZE` square preserving aspect ratio
(black-padded), `_to_blob` converts BGR→RGB and normalizes to NCHW, and the
ONNX session (`CUDAExecutionProvider` falling back to `CPUExecutionProvider`)
returns raw YOLO predictions that `_postprocess` filters by
`DETECTOR.CONF_THRESHOLD`, runs through `cv2.dnn.NMSBoxes` at
`DETECTOR.IOU_THRESHOLD`, and un-letterboxes back into the cropped image's
pixel space. Every configured class visible in the frame is kept, not just
the one the current scan is looking for — an object waiting its turn stays
tracked so it can be picked without a second scan once it's up.

### Step 3 — Pose estimation (`PoseEstimator`, [`src/vision/pose_estimator.py`](src/vision/pose_estimator.py))

For each kept detection, `PoseEstimator.predict()` builds four inputs in the
model's fixed contract order: the full agentview frame resized to
`POSE.POS_IMAGE_SIZE` (xyz stream context), normalized bbox coordinates plus
derived area/center, a one-hot class vector, and the detection's own crop
resized to `POSE.ROTATION_IMAGE_SIZE` (rotation stream detail) — falling
back to the full frame if the crop is degenerate. The ONNX session returns a
camera-frame `xyz` and a 6D rotation vector, which
`PoseEstimator.rot6d_to_matrix` reconstructs into an orthonormal 3×3
rotation matrix via Gram-Schmidt (Zhou et al., 2019 representation, matching
the training convention in
[`dev-robosuite-pickplace-vision`](../dev-robosuite-pickplace-vision)).

### Step 4 — Coordinate transform (`CoordinateTransformer`, [`src/geometry/coordinate_transform.py`](src/geometry/coordinate_transform.py))

`camera_to_world_frame(xyz_cam, rot_cam, cam_xpos, cam_xmat)` computes
`world_xpos = cam_xmat @ xyz_cam + cam_xpos` and `world_xmat = cam_xmat @
rot_cam` — the exact algebraic inverse of the vision repo's
`world_to_camera_frame` (`cam_xmat.T @ (...)`), since `cam_xmat` is an
orthonormal rotation matrix and its transpose is its inverse: one repo
strips the camera transform out to build training targets, this one puts it
back to turn predictions into robot-frame poses. It then permutes
`world_xmat`'s columns into the "green axis" grasp-alignment convention used
at training-label time, extracts a yaw angle via
`scipy.spatial.transform.Rotation`, and wraps it into `[-90°, 90°]` — a 180°
turn about the vertical axis is an equivalent grasp for all four object
classes, so yaw only needs resolving modulo 180°, not 360°. The camera's own
world position/rotation (`cam_xpos`/`cam_xmat`) are read once from
`env.sim.data.cam_xpos`/`cam_xmat` at executor init and reused for every
scan, since the camera itself never moves.

### Step 5 — Control (`PickPlaceController`, `JointMotion`, [`src/control/`](src/control/))

`PickPlaceController.run()` drives one object through an 11-stage state
machine (`Stage`, [`src/control/stage.py`](src/control/stage.py)) — the
execution order lives in `run()`'s stage transitions, not in `Stage`'s
numeric values (kept only for log compatibility with earlier experiments):

1. **RAISE** — rise to a shared safe travel height above the table.
2. **ALIGN_XY** — move horizontally to the detected object, still at travel height.
3. **DESCEND** — drop to just above the object. `Can` (cylindrical, rotationally symmetric) skips straight to `FINE_DESCEND`; every other class goes through `ORIENT_YAW` first.
4. **ORIENT_YAW** — rotate joint 7 only, toward the detected grasp yaw folded to the nearest 180°-symmetric equivalent, so the wrist never takes the long way round.
5. **FINE_DESCEND** — the final, slower approach to contact height, using a small per-class Z correction that compensates for each mesh's real contact point vs. its reported origin.
6. **GRASP** — hold pose, close the gripper, and verify contact.
7. **LIFT** — rise back to travel height. Convergence requires the object to still be grasped *and* to have risen a minimum height — catching a "grasp" that's really just resting between the fingers.
8. **REZERO_YAW** — a second joint-7-only stage that re-zeros the wrist to a canonical orientation before transport, regardless of what grasp yaw stage 4 needed.
9. **MOVE_TO_BIN** — travel to the target bin at travel height, holding the re-zeroed orientation.
10. **DESCEND_TO_BIN** — drop to a per-class absolute release height inside the bin, at a tighter tolerance than transit stages.
11. **RELEASE** — hold pose and open the gripper.

Every Cartesian stage also carries a wrist target and only advances once
both position *and* wrist orientation have converged; a stage stuck for
`STAGES.MAX_TICKS_PER_STAGE` ticks raises rather than correcting forever.

`JointMotion.action()` ([`src/control/joint_motion.py`](src/control/joint_motion.py))
turns one Cartesian pose command into a 7-joint absolute-position action
every tick, always as one small bounded step from the robot's *current*
measured joints — it never teleports. Position error is proportionally
scaled and clamped (`MOTION.MAX_POSITION_DELTA`, rescaling the XY vector
rather than clipping each axis independently, to keep multi-axis moves on a
straight line); orientation correction is off by default and only active
for stages that explicitly hold world yaw. The Cartesian step itself comes
from a **damped-least-squares differential IK** solve (`mujoco.mj_jacSite`
at the gripper site, damping = `MOTION.IK_DAMPING`) restricted to position +
yaw — roll/pitch are deliberately excluded from this Jacobian so they never
fight the separate **J5/J6 wrist solve**, which every tick re-derives the
joint angles that keep the tool's Z-axis pointing straight down via local
forward-kinematics optimization (SciPy `least_squares` over J5+J6 jointly,
or a 1-D fallback if J5 isn't configured), blended in through a
feedforward-plus-proportional tracking correction so the joint can keep up
with a wrist target that shifts every tick. Yaw-only stages
(`ORIENT_YAW`/`REZERO_YAW`) instead hold every other joint at its current
position and step only joint 7, so a yaw correction can never accidentally
translate the gripper off the object.

## Recording (`BaseRunObserver`)

Recording is entirely decoupled from the pick-and-place logic:
`PickPlaceExecutor` only reports what happens, through five lifecycle hooks
defined by the `BaseRunObserver` ABC
([`src/recording/run_observer.py`](src/recording/run_observer.py)) —
`on_run_start` (once, after `env.reset()`), `on_scan`/`on_target_detected`
(per detection scan, before every object), `on_step` (after every
controller tick), and `on_run_end` (once, in a `finally`, so a crashed or
early-terminated run still gets whatever it captured).
[`src/recording/factory.py`](src/recording/factory.py)'s `build_observers()`
reads `VisualizationConfig` and constructs any subset of the three
observers below — each ignores the hooks it has no use for.

### Video (`VideoObserver`, [`src/recording/video_observer.py`](src/recording/video_observer.py))

Set `VISUALIZATION.VIDEO_ENABLED` to `true`, then run the headless command
above. Two Full-HD (1920×1080) MP4s are written by default —
`outputs/pickplace_agentview.mp4` and `outputs/pickplace_frontview.mp4` —
downsampled to one captured tick every `VIDEO_CAPTURE_EVERY_TICKS` (6 by
default) to bound render time, using a fast H.264 preset. The detection
overlay (box, label, red-X/green-Y/blue-Z pose axes) is only drawn during
the pre-grasp approach stages (`RAISE` through `FINE_DESCEND`) — the overlay
marks where the scan *detected* the object, which stops being accurate the
moment the gripper starts moving it. The front-view box has no detector of
its own behind it; instead it comes from a MuJoCo segmentation render of the
object's visual geometry, so occlusion by the bin walls resolves correctly
for free.

### Trajectory (`TrajectoryObserver`, [`src/recording/trajectory_observer.py`](src/recording/trajectory_observer.py))

Set `VISUALIZATION.TRAJECTORY_ENABLED` to `true` to export a compact
recording for the [local simulation website](#local-simulation-website). It
writes three files derived from `TRAJECTORY_PATH`/`TRAJECTORY_MODEL_PATH`: a
packaged `model.zip` (the run's MJCF scene, exported at `on_run_start` so a
browser can render it within seconds, well before the run finishes), an
append-only `live.bin` (fixed-stride binary records —
`time:f32, qpos:f32[nq], object:u8, stage:u8, pad:u8[2]` — flushed after
every tick, so a concurrent reader sees growth without waiting for the
process to exit), and the final `trajectory.bin` (a compact
length-prefixed-header binary, written once at `on_run_end`). Every recorded
frame is the complete MuJoCo `qpos` vector, which is why the browser player
never re-runs the controller or physics — replay is just re-applying
recorded `qpos` frames against the packaged scene.

### Scan images (`ScanImageObserver`, [`src/recording/scan_image_observer.py`](src/recording/scan_image_observer.py))

`VISUALIZATION.SCAN_IMAGES_ENABLED` is on by default. Every detection scan
saves an annotated `outputs/scans/scan_NN_agentview.png` (box, label, and
pose axes for every detected object, not just the one currently sought), and
the very first scan also saves an unannotated `outputs/scans/raw_agentview.png`
of the opening scene, for documentation. Set `SCAN_IMAGES_DIR` to change
where they're written.

## Local simulation website

The browser player replays exact recorded MuJoCo states while the Python API
runs new headless simulations. It downloads the shared scene model once and
caches it; each later run transfers only a compact trajectory binary.

Start the API in one terminal:

```bash
uv run uvicorn src.web.app:app --reload
```

Start the player in a second terminal:

```bash
cd src/web/player
npm install
npm run dev
```

Open Vite's local address and select **Run new simulation**.

### Web API (`SimulationJobs`, [`src/web/`](src/web/))

[`app.py`](src/web/app.py) is a thin FastAPI layer over one module-level
`SimulationJobs` registry ([`simulation_jobs.py`](src/web/simulation_jobs.py))
— every route just delegates to it. `POST /api/simulations` refuses with
`409` while another run is in progress (only one headless simulation runs at
a time), otherwise writes `request.json` into a fresh
`outputs/web/runs/<job_id>/` directory and spawns
[`worker.py`](src/web/worker.py) as a **detached subprocess**
(`python -m src.web.worker <request.json>`, its own process group, so it
survives the API process restarting). The worker runs the same `main.py`
pipeline headlessly with trajectory export forced on and MP4 recording
forced off, then writes `result.json` on success or `error.json` on any
exception.

`SimulationJobs.refresh()` derives a job's status purely from what's on disk
each time it's called — `result.json` → `completed`, `error.json` →
`failed`, the tracked process still alive → `running` or `paused` (depending
on whether a `paused.flag` file exists in the run directory), otherwise
`stopped` or a hard-crash `failed`. Pause/resume/stop are implemented as
filesystem signals: `POST .../pause` touches `paused.flag`, which
`PickPlaceExecutor._wait_while_paused` polls between controller ticks;
`POST .../stop` terminates the worker process and keeps whatever it streamed
so far. `GET .../live` and `GET .../live/frames?from=N` expose the same
fixed-stride `live.bin` the trajectory observer writes, letting the player
show a run *while it's still in progress* — `/live/frames` only ever serves
whole frames, so a partially-written trailing record from a concurrently
writing worker is never sent. Run folders older than a day (by directory
mtime) are deleted automatically the next time a simulation is started.

### Browser player

A Vite project with no UI framework — a single JS module
([`src/main.js`](src/web/player/src/main.js)) drives Three.js rendering,
MuJoCo-WASM model compilation from the downloaded scene ZIP, and the FastAPI
calls directly. Live progress is polled (`GET .../live` every 40 ms), not
pushed over a socket; the player smooths the resulting stream by
exponentially easing its displayed simulation time toward the newest
arrived frame, rather than trying to play frames at a fixed rate the
headless run doesn't actually have. It also supports a fully
static/serverless mode (`?static=1&preset=...`) and an `?embed=1` +
`postMessage` remote-control contract for hosting the player inside another
page (used by the portfolio demo in `docs/`). See
[src/web/player/README.md](src/web/player/README.md) for the full
pause/resume/stop and replay-controls walkthrough.

## Project structure

```
dev-robot-pickplace-control/
├── config/
│   ├── config.yaml          # detector/pose/environment/motion/stage tuning (see above)
│   └── logging.yaml         # stdlib logging.dictConfig
├── models/
│   ├── yolo_detector.onnx   # copied from dev-robosuite-pickplace-vision
│   └── pose_estimator.onnx  # copied from dev-robosuite-pickplace-vision
├── docs/                    # GitHub Pages portfolio demo (static build + sample run)
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
    │   ├── trajectory_recorder.py       # portable model + qpos recording for web playback
    │   ├── trajectory_observer.py       # drives TrajectoryRecorder from executor hooks
    │   ├── video_recorder.py            # agentview + frontview MP4 writer
    │   ├── video_observer.py            # drives VideoRecorder, adds detection/pose overlays
    │   └── scan_image_observer.py       # saves an annotated PNG per detection scan
    ├── utils/
    │   ├── logging_configurator.py      # logging.yaml-driven setup
    │   ├── system_configurator.py       # ConfigReader/ConfigAssembler/load_config()
    │   └── types.py                     # shared configuration and state data
    ├── vision/
    │   ├── onnx_detector.py             # YOLO ONNX inference
    │   └── pose_estimator.py            # pose_estimator.onnx inference
    └── web/
        ├── app.py                       # FastAPI routes: queue runs, serve trajectories/model
        ├── simulation_jobs.py           # job registry: state, run dirs, disk cleanup
        ├── worker.py                    # subprocess entrypoint for one queued run
        └── player/                      # browser viewer (Vite + Three.js + MuJoCo WASM)
```

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
