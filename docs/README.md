# Trajectory Demo Player

A browser-based viewer that replays a recorded pick-and-place run frame by
frame, using a saved trajectory and 3D scene, without needing to run the
simulation or controller itself.

This folder is served via GitHub Pages.

## Project structure

```
docs/
├── index.html                     # the viewer
└── assets/
    ├── script.js                      # viewer bundle
    ├── style.css                      # viewer styles
    ├── mujoco.wasm                    # MuJoCo physics engine, compiled to WebAssembly
    ├── sample-trajectory.bin          # one recorded run's qpos frames
    ├── sample-trajectory.result.json  # that run's placement/vision-accuracy summary
    └── scene-model.zip                # packaged MJCF scene model
```

## Regenerating the demo

One command produces a fresh run and writes straight into `docs/assets/`,
since `config.yaml` already points `TRAJECTORY_PATH`/`TRAJECTORY_MODEL_PATH`
there:

```bash
# macOS
uv run mjpython main.py

# Windows / Linux
uv run python main.py
```

This is all you need for a new simulation. The viewer itself
(`index.html`, `script.js`, `style.css`, `mujoco.wasm`) doesn't change, only
the recorded data it plays back.
