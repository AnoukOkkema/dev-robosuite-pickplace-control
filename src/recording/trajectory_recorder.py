"""Save a cached MuJoCo model and compact browser-playback trajectories."""

from __future__ import annotations

import json
import logging
import struct
import xml.etree.ElementTree as element_tree
import zipfile
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

from src.control.stage import Stage


class TrajectoryRecorder:
    """Record every simulation state of a run for the browser player.

    After every controller step the recorder stores the full ``qpos`` vector:
    the positions of all arm joints, the gripper, and the loose objects. The
    player replays a run by simply re-applying those positions frame by
    frame, so it never has to run the controller or the physics itself.

    It writes three files:

    * ``model.zip`` (:meth:`export_model`): the packaged 3D scene, which the
      player downloads once in order to render anything at all.
    * ``live.bin`` (:meth:`start` + :meth:`capture`): a stream that grows
      while the run is still going, so the player can show the robot moving
      live.
    * ``trajectory.bin`` (:meth:`export`): the complete recording, written
      once the run is over, used for replays.
    """

    def __init__(
        self,
        output_path: str,
        model_path: str,
        logger: logging.Logger | None = None,
    ) -> None:
        """Create an empty trajectory recorder.

        Args:
            output_path: Destination of ``trajectory.bin``.
            model_path: Destination of ``model.zip``.
            logger: Logger used for export progress.
        """
        self.output_path = Path(output_path)
        self.model_path = Path(model_path)
        self.logger = logger or logging.getLogger(__name__)
        self._frames: list[dict[str, Any]] = []
        self._live_path = self.output_path.with_name("live.bin")
        self._live_meta_path = self.output_path.with_name("live.meta.json")
        self._live_file: BinaryIO | None = None
        self._live_object_codes: dict[str, int] = {}
        self._live_stage_codes: dict[str, int] = {}

    def start(self, nq: int, object_names: list[str]) -> None:
        """Open the live-progress stream for one run.

        Call once, as soon as the compiled model's ``nq`` and the run's
        object order are known -- before the first :meth:`capture` -- so a
        browser polling ``live.bin`` can render seconds into a run instead
        of only after :meth:`export` at the very end. Object and stage names
        must be fixed up front because ``live.bin`` is append-only: unlike
        the final trajectory, there is no second pass to collect them.

        Args:
            nq: Number of generalized-position values in every frame.
            object_names: Object classes in this run's placement order.
        """
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        stage_names = [stage.name for stage in Stage]
        self._live_object_codes = {
            name: index + 1 for index, name in enumerate(object_names)
        }
        self._live_stage_codes = {
            name: index + 1 for index, name in enumerate(stage_names)
        }
        self._live_meta_path.write_text(
            json.dumps(
                {
                    "format_version": 1,
                    "nq": int(nq),
                    "object_names": object_names,
                    "stage_names": stage_names,
                },
                separators=(",", ":"),
            )
        )
        self._live_file = self._live_path.open("wb")

    def capture(
        self,
        simulation_time: float,
        qpos: np.ndarray,
        object_name: str | None = None,
        stage_name: str | None = None,
    ) -> None:
        """Append one complete simulation state to the recording.

        Args:
            simulation_time: MuJoCo simulation time in seconds.
            qpos: Current generalized-position vector from MuJoCo.
            object_name: Object currently handled by the controller, if any.
            stage_name: Active pick-and-place stage, if any.
        """
        frame = {
            "time": float(simulation_time),
            "qpos": np.asarray(qpos, dtype=float).tolist(),
        }
        if object_name is not None:
            frame["object_name"] = object_name
        if stage_name is not None:
            frame["stage_name"] = stage_name
        self._frames.append(frame)
        if self._live_file is not None:
            self._append_live_frame(frame)

    def _append_live_frame(self, frame: dict[str, Any]) -> None:
        """Write one fixed-stride record to the live stream and flush it.

        The record layout is ``time:f32, qpos:f32[nq], object:u8, stage:u8,
        pad:u8[2]`` -- 4-byte aligned throughout, so a reader can decode any
        whole number of trailing records straight into ``Float32Array``s.
        """
        qpos_bytes = np.asarray(frame["qpos"], dtype=np.float32).tobytes()
        object_code = self._live_object_codes.get(frame.get("object_name"), 0)
        stage_code = self._live_stage_codes.get(frame.get("stage_name"), 0)
        self._live_file.write(
            struct.pack("<f", frame["time"])
            + qpos_bytes
            + struct.pack("<BBxx", object_code, stage_code)
        )
        # Push to the OS page cache immediately so concurrent readers (the
        # web API) observe growth without waiting for this process to exit.
        self._live_file.flush()

    def finish_live(self) -> None:
        """Close the live stream. Safe to call even if :meth:`start` was not."""
        if self._live_file is not None:
            self._live_file.close()
            self._live_file = None

    def export_model(self, model) -> None:
        """Package the reusable MJCF ZIP as soon as the model is known.

        Call this right after the model compiles so a browser player can
        show the scene immediately, rather than waiting for the whole run to
        finish. Idempotent: skips work if the ZIP already exists, and safe
        to leave uncalled since :meth:`export` also packages the model.

        Args:
            model: Robosuite model wrapper used during the recorded run.

        Raises:
            RuntimeError: If model XML cannot be converted into a portable ZIP.
        """
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        if self.model_path.exists():
            return
        try:
            self._package_model(model.get_xml())
        except Exception as error:
            raise RuntimeError(
                "Could not package the MuJoCo model for trajectory playback."
            ) from error

    def export(self, model) -> None:
        """Write the trajectory binary and a reusable self-contained MJCF archive.

        Args:
            model: Robosuite model wrapper used during the recorded run.
        """
        self.finish_live()
        if not self._frames:
            self.logger.warning("TRAJECTORY | No states captured; export skipped.")
            return

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_binary_trajectory(model.nq)
        self.export_model(model)

        self.logger.info(
            "TRAJECTORY | Saved %d compact states | trajectory=%s | model=%s",
            len(self._frames),
            self.output_path,
            self.model_path,
        )

    def _write_binary_trajectory(self, nq: int) -> None:
        """Write qpos, time, and stage metadata in a browser-friendly binary format.

        JSON stores every floating-point value as text and repeats object and
        stage names for every frame. The binary layout keeps those names once
        in a small header and stores numeric state arrays directly afterwards.

        Args:
            nq: Number of generalized-position values in every recorded frame.
        """
        object_names = self._unique_frame_values("object_name")
        stage_names = self._unique_frame_values("stage_name")
        object_codes = {name: index + 1 for index, name in enumerate(object_names)}
        stage_codes = {name: index + 1 for index, name in enumerate(stage_names)}
        times = np.asarray([frame["time"] for frame in self._frames], dtype=np.float32)
        qpos = np.asarray([frame["qpos"] for frame in self._frames], dtype=np.float32)
        object_frames = np.asarray(
            [object_codes.get(frame.get("object_name"), 0) for frame in self._frames],
            dtype=np.uint8,
        )
        stage_frames = np.asarray(
            [stage_codes.get(frame.get("stage_name"), 0) for frame in self._frames],
            dtype=np.uint8,
        )
        if qpos.shape != (len(self._frames), nq):
            raise ValueError("Recorded qpos states do not match the MuJoCo model")

        header = json.dumps(
            {
                "format_version": 3,
                "frame_count": len(self._frames),
                "nq": int(nq),
                "object_names": object_names,
                "stage_names": stage_names,
                "layout": ["time_f32", "qpos_f32", "object_u8", "stage_u8"],
            },
            separators=(",", ":"),
        ).encode("utf-8")
        padding = b"\0" * ((-4 - len(header)) % 4)
        with self.output_path.open("wb") as trajectory_file:
            trajectory_file.write(struct.pack("<I", len(header)))
            trajectory_file.write(header)
            trajectory_file.write(padding)
            trajectory_file.write(times.tobytes())
            trajectory_file.write(qpos.tobytes())
            trajectory_file.write(object_frames.tobytes())
            trajectory_file.write(stage_frames.tobytes())

    def _unique_frame_values(self, key: str) -> list[str]:
        """Return encountered non-empty values for one frame key in order."""
        return list(
            dict.fromkeys(
                frame[key]
                for frame in self._frames
                if isinstance(frame.get(key), str) and frame[key]
            )
        )

    def _package_model(self, model_xml: str) -> None:
        """Write generated MJCF and every referenced Robosuite asset to ZIP.

        ``MjModel.get_xml()`` returns asset paths from the local Robosuite
        installation. MuJoCo's standard spec ZIP does not retain those files,
        so browser playback needs this explicit portable package.

        Args:
            model_xml: Generated MJCF XML from the compiled simulation model.

        Raises:
            FileNotFoundError: If an asset referenced by generated MJCF is absent.
        """
        root = element_tree.fromstring(model_xml)
        compiler = root.find("compiler")
        if compiler is not None:
            # Every asset path below is made relative to scene.xml.
            compiler.set("meshdir", "")
            compiler.set("texturedir", "")

        assets: dict[Path, str] = {}
        for element in root.iter():
            source_name = element.get("file")
            if not source_name:
                continue
            source_path = Path(source_name).expanduser().resolve()
            if not source_path.is_file():
                raise FileNotFoundError(f"MJCF asset does not exist: {source_path}")
            archive_name = assets.setdefault(
                source_path,
                f"assets/{len(assets):03d}_{source_path.name}",
            )
            element.set("file", archive_name)

        with zipfile.ZipFile(self.model_path, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(
                "scene.xml",
                element_tree.tostring(root, encoding="unicode"),
            )
            for source_path, archive_name in assets.items():
                archive.write(source_path, archive_name)
