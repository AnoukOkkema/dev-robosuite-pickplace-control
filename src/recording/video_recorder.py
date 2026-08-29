import logging
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

from src.util.types import ImageSize


class VideoRecorder:
    """Write one Full-HD MP4 recording per camera during a robot run.

    The agent view records object detections and pose axes, while the front
    view records the full Panda motion. Every call to :meth:`capture` writes
    one frame to both files at once, so the agent-view and front-view MP4s
    always have the same frame count and never drift out of sync with each
    other.
    """

    def __init__(
        self,
        output_path: str,
        fps: int,
        capture_every_ticks: int,
        logger=None,
    ) -> None:
        """Initialize unopened writers for the two camera recordings.

        Args:
            output_path: Base path. ``pickplace.mp4`` becomes separate
                ``pickplace_agentview.mp4`` and ``pickplace_frontview.mp4`` files.
            fps: Playback frames per second.
            capture_every_ticks: Controller steps between recorded frames.
            logger: Logger used to report completed recordings.

        Raises:
            ValueError: If ``fps`` or ``capture_every_ticks`` is not positive.
        """
        if fps <= 0 or capture_every_ticks <= 0:
            raise ValueError("Video fps and capture interval must be positive")

        base_path = Path(output_path)
        self.agentview_path = base_path.with_name(f"{base_path.stem}_agentview.mp4")
        self.frontview_path = base_path.with_name(f"{base_path.stem}_frontview.mp4")
        self.fps = fps
        self.capture_every_ticks = capture_every_ticks
        self.logger = logger or logging.getLogger(__name__)
        self._agentview_writer = None
        self._frontview_writer = None
        self._frame_count = 0
        self._controller_tick_count = 0

        # Output canvas of both MP4s; source frames are letterboxed onto it.
        self.frame_size = ImageSize(height=1080, width=1920)
        # Human-readable caption per stage, shown in the video's status line.
        # "DETECTIONS" is the scan phase reported by run(), not a Stage.
        self.stage_labels = {
            "DETECTIONS": "Scanning the scene",
            "RAISE": "Raising to safe height",
            "ALIGN_XY": "Moving above object",
            "DESCEND": "Descending to object",
            "ORIENT_YAW": "Aligning gripper yaw",
            "FINE_DESCEND": "Final grasp approach",
            "GRASP": "Closing gripper",
            "LIFT": "Lifting object",
            "REZERO_YAW": "Setting transport yaw",
            "MOVE_TO_BIN": "Moving above bin",
            "DESCEND_TO_BIN": "Descending into bin",
            "RELEASE": "Opening gripper",
        }

    def should_capture_motion_frame(self) -> bool:
        """Return whether the current controller step should be rendered.

        The controller ticks far more often than a demo video needs frames,
        so recording every tick would make the export excessively long.
        Sampling every ``capture_every_ticks``-th tick instead keeps the
        video watchable while still covering the run's motion evenly.
        """
        self._controller_tick_count += 1
        return self._controller_tick_count % self.capture_every_ticks == 0

    @property
    def frame_count(self) -> int:
        """Return the number of matching camera-frame pairs written so far."""
        return self._frame_count

    def capture(
        self,
        agentview_bgr: np.ndarray,
        frontview_rgb: np.ndarray,
        object_name: str,
        stage_name: str,
    ) -> None:
        """Append matching, labelled frames to the two MP4 recordings.

        ``stage_name`` accepts either a plain string or a ``Stage`` enum
        because the two observers that call this hand it over differently:
        one passes ``stage.name``, the other passes the raw enum. The label
        lookup below normalizes whichever form arrives.

        Args:
            agentview_bgr: Full BGR agent-view image, optionally annotated.
            frontview_rgb: Full RGB front-view image.
            object_name: Object currently being detected or moved.
            stage_name: Current pick-and-place stage name or ``Stage`` enum.
        """
        stage_key = getattr(stage_name, "name", None)
        if not isinstance(stage_key, str):
            stage_key = str(stage_name)
        stage_label = self.stage_labels.get(
            stage_key,
            stage_key.replace("_", " ").title(),
        )
        status = f"Object: {object_name}  |  {stage_label}"
        agentview_frame = self._prepare_frame(
            agentview_bgr,
            view_name="Agent view",
            status=status,
        )
        frontview_frame = self._prepare_frame(
            cv2.cvtColor(frontview_rgb, cv2.COLOR_RGB2BGR),
            view_name="Front view",
            status=status,
        )
        agentview_writer, frontview_writer = self._writers_for_frame()
        agentview_writer.append_data(cv2.cvtColor(agentview_frame, cv2.COLOR_BGR2RGB))
        frontview_writer.append_data(cv2.cvtColor(frontview_frame, cv2.COLOR_BGR2RGB))
        self._frame_count += 1

    def close(self) -> None:
        """Finish both MP4 files and log their paths when frames were recorded."""
        if self._agentview_writer is None:
            return
        self._agentview_writer.close()
        self._frontview_writer.close()
        self._agentview_writer = None
        self._frontview_writer = None
        self.logger.info(
            "VIDEO | Saved %d frames | agentview=%s | frontview=%s",
            self._frame_count,
            self.agentview_path,
            self.frontview_path,
        )

    def _writers_for_frame(self):
        """Create compatible H.264 MP4 writers before the first captured frame."""
        if self._agentview_writer is None:
            self.agentview_path.parent.mkdir(parents=True, exist_ok=True)
            writer_options = {
                "fps": self.fps,
                "codec": "libx264",
                "pixelformat": "yuv420p",
                "ffmpeg_params": ["-preset", "ultrafast", "-crf", "23"],
                # 1080 is not divisible by 16; keep the requested 1920x1080
                # instead of letting ImageIO silently resize it to 1920x1088.
                "macro_block_size": 1,
            }
            self._agentview_writer = imageio.get_writer(
                self.agentview_path,
                **writer_options,
            )
            self._frontview_writer = imageio.get_writer(
                self.frontview_path,
                **writer_options,
            )
        return self._agentview_writer, self._frontview_writer

    def _prepare_frame(
        self,
        image_bgr: np.ndarray,
        view_name: str,
        status: str,
    ) -> np.ndarray:
        """Create a 1920x1080 frame with a readable two-line text overlay.

        Camera frames don't natively match the fixed output canvas's aspect
        ratio. Stretching them would distort the scene, and plain padding
        would leave dead black bars, so the frame is instead centered at its
        native aspect ratio over a heavily blurred, stretched copy of
        itself, so the empty space still reads as part of the scene. Each
        caption is drawn twice, an offset light shadow then the color text
        on top, so it stays legible over both light and dark backgrounds.

        Args:
            image_bgr: Source BGR camera frame to letterbox onto the fixed
                1920x1080 canvas.
            view_name: Camera label drawn as the frame's first caption line.
            status: Object-and-stage caption drawn as the frame's second line.
        """
        height, width = image_bgr.shape[:2]
        background = cv2.resize(
            image_bgr, (self.frame_size.width, self.frame_size.height)
        )
        background = cv2.GaussianBlur(background, (0, 0), sigmaX=35)

        scale = min(self.frame_size.width / width, self.frame_size.height / height)
        foreground_size = (round(width * scale), round(height * scale))
        foreground = cv2.resize(image_bgr, foreground_size)
        frame = background
        offset_x = (self.frame_size.width - foreground_size[0]) // 2
        offset_y = (self.frame_size.height - foreground_size[1]) // 2
        frame[
            offset_y : offset_y + foreground_size[1],
            offset_x : offset_x + foreground_size[0],
        ] = foreground

        text_color = (42, 32, 192)  # #c0202a in BGR for OpenCV.
        shadow_color = (245, 245, 245)
        text_x = 34
        cv2.putText(
            frame,
            view_name,
            (text_x + 2, 57),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.4,
            shadow_color,
            5,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            view_name,
            (text_x, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.4,
            text_color,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            status,
            (text_x + 2, 109),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            shadow_color,
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            status,
            (text_x, 107),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            text_color,
            1,
            cv2.LINE_AA,
        )
        return frame
