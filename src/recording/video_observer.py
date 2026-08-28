import logging

import cv2
import numpy as np

from src.control.stage import Stage
from src.recording.run_observer import BaseRunObserver
from src.recording.video_recorder import VideoRecorder


class VideoObserver(BaseRunObserver):
    """Record an annotated agent-view and front-view MP4 by listening to the executor.

    The overlay uses the pose from the most recent :meth:`on_target_detected`
    call, so it always shows the object the run is currently approaching.
    """

    def __init__(
        self, video_path: str, fps: int, capture_every_ticks: int, logger=None
    ) -> None:
        """Create the observer and its underlying recorder.

        Args:
            video_path: Base path passed to :class:`VideoRecorder`.
            fps: Playback frames per second.
            capture_every_ticks: Controller steps between recorded frames.
            logger: Logger used to report the finished MP4 files.
        """
        self._recorder = VideoRecorder(
            video_path,
            fps,
            capture_every_ticks,
            logger=logger or logging.getLogger(__name__),
        )
        self._executor = None
        self._front_cam_xpos = None
        self._front_cam_xmat = None
        self._active_target_pose = None
        self._skip_first_frame = False
        # Only during these approach stages does the recorded MP4 get the
        # detection overlay (bounding box + pose axes). The overlay is drawn
        # at the position where the scan detected the object. That is correct
        # while the object still lies untouched on that spot, but from GRASP
        # onward the gripper moves the object while the box would stay behind
        # at the old position.
        self._annotated_stages = frozenset(
            {
                Stage.RAISE,
                Stage.ALIGN_XY,
                Stage.DESCEND,
                Stage.ORIENT_YAW,
                Stage.REALIGN_XY,
                Stage.FINE_DESCEND,
            }
        )

    def on_run_start(self, executor) -> None:
        """Resolve the front camera and arm the first-frame skip.

        The reset state contains newly placed objects before the first
        simulation update, so the first captured frame is skipped. It
        would otherwise show objects still settling into view.
        """
        self._executor = executor
        self._front_cam_xpos, self._front_cam_xmat = (
            executor._resolve_camera_extrinsics("frontview")
        )
        self._active_target_pose = None
        self._skip_first_frame = True

    def on_scan(self, agentview_frame: np.ndarray, poses_cam: list) -> None:
        """Unused. The video overlay is driven by ``on_target_detected`` instead."""

    def on_target_detected(self, class_name: str, pose_cam: tuple) -> None:
        """Remember the detected pose so the next frames can be annotated with it."""
        self._active_target_pose = pose_cam

    def on_step(self, object_name: str, stage: Stage) -> None:
        """Render and append one labelled frame pair, downsampled by the recorder."""
        if not self._recorder.should_capture_motion_frame():
            return
        if self._skip_first_frame:
            self._skip_first_frame = False
            return

        agentview_bgr = cv2.cvtColor(
            self._executor._render_camera_frame("agentview"), cv2.COLOR_RGB2BGR
        )
        frontview_rgb = self._executor._render_camera_frame("frontview")
        if stage in self._annotated_stages and self._active_target_pose is not None:
            agentview_bgr = self._annotate_agentview(
                agentview_bgr, self._active_target_pose
            )
            frontview_bgr = cv2.cvtColor(frontview_rgb, cv2.COLOR_RGB2BGR)
            frontview_bgr = self._annotate_frontview(
                frontview_bgr, self._active_target_pose
            )
            frontview_rgb = cv2.cvtColor(frontview_bgr, cv2.COLOR_BGR2RGB)

        self._recorder.capture(agentview_bgr, frontview_rgb, object_name, stage)

    def on_run_end(self) -> None:
        """Finish both MP4 files."""
        self._recorder.close()

    # ------------------------------------------------------------------
    # Annotation
    # ------------------------------------------------------------------

    def _annotate_agentview(
        self, agentview_bgr: np.ndarray, pose_cam: tuple
    ) -> np.ndarray:
        """Draw the active target's box, label, and pose axes in agent view."""
        class_name, box, xyz_cam, rot_cam = pose_cam
        crop_region = self._executor.crop_region
        x1, y1, x2, y2 = (int(round(value)) for value in box)
        full_frame_box = (
            x1 + crop_region.x1,
            y1 + crop_region.y1,
            x2 + crop_region.x1,
            y2 + crop_region.y1,
        )
        annotated = self._draw_pose_axes(
            agentview_bgr, full_frame_box, xyz_cam, rot_cam, camera_name="agentview"
        )
        cv2.putText(
            annotated,
            class_name,
            (full_frame_box[0], max(full_frame_box[1] - 5, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return annotated

    def _annotate_frontview(
        self, frontview_bgr: np.ndarray, pose_cam: tuple
    ) -> np.ndarray:
        """Draw the active target's projected box, label, and pose axes, front view."""
        class_name, _, xyz_agent, rot_agent = pose_cam
        executor = self._executor
        world_position = executor.cam_xmat @ xyz_agent + executor.cam_xpos
        world_rotation = executor.cam_xmat @ rot_agent
        front_position = self._front_cam_xmat.T @ (
            world_position - self._front_cam_xpos
        )
        front_rotation = self._front_cam_xmat.T @ world_rotation
        front_box = self._visible_object_box(class_name)

        annotated = self._draw_pose_axes(
            frontview_bgr,
            front_box,
            front_position,
            front_rotation,
            camera_name="frontview",
        )
        if front_box is not None:
            cv2.putText(
                annotated,
                class_name,
                (front_box[0], max(front_box[1] - 5, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        return annotated

    def _visible_object_box(self, class_name: str) -> tuple[int, int, int, int] | None:
        """Return the front-view box of the visible part of an object.

        The front camera has no detector of its own. MuJoCo segmentation gives
        the geometry identifier for every visible pixel, so table walls and
        other foreground meshes naturally occlude the target box.
        """
        env = self._executor.env
        visual_geom_id = env.sim.model.geom_name2id(f"{class_name}_g0_visual")
        segmentation = self._render_segmentation_frame("frontview")
        visible_pixels = np.argwhere(segmentation[:, :, 0] == visual_geom_id)
        if visible_pixels.size == 0:
            return None

        y_values, x_values = visible_pixels.T
        return (
            int(x_values.min()),
            int(y_values.min()),
            int(x_values.max()),
            int(y_values.max()),
        )

    def _render_segmentation_frame(self, camera_name: str) -> np.ndarray:
        """Render visible geometry identifiers for one MuJoCo camera frame."""
        executor = self._executor
        executor._render_camera_frame(camera_name)  # ensures the renderer exists
        renderer = executor._renderers[camera_name]
        renderer.enable_segmentation_rendering()
        try:
            renderer.update_scene(
                executor.env.sim.data._data,
                camera=camera_name,
                scene_option=executor._scene_option,
            )
            return renderer.render()
        finally:
            renderer.disable_segmentation_rendering()

    def _draw_pose_axes(
        self,
        frame: np.ndarray,
        box,
        xyz_cam: np.ndarray,
        rot_cam: np.ndarray,
        camera_name: str,
        axis_length: float = 0.05,
    ) -> np.ndarray:
        """Draw a bounding box and red-X, green-Y, blue-Z pose axes."""
        annotated = frame.copy()
        axis_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]

        if box is not None:
            x1, y1, x2, y2 = (int(round(value)) for value in box)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 1)

        origin = self._project_point(xyz_cam, camera_name)
        if origin is None:
            return annotated

        ox, oy = origin
        cv2.circle(annotated, (int(ox), int(oy)), 4, (255, 255, 255), -1)
        for axis_index, color in enumerate(axis_colors):
            axis_point_cam = xyz_cam + rot_cam[:, axis_index] * axis_length
            axis_pixel = self._project_point(axis_point_cam, camera_name)
            if axis_pixel is None:
                continue
            ax, ay = axis_pixel
            cv2.line(
                annotated, (int(ox), int(oy)), (int(ax), int(ay)), color, 2, cv2.LINE_AA
            )
        return annotated

    def _project_point(self, xyz_cam: np.ndarray, camera_name: str):
        """Project a camera-frame point into full-frame pixel coordinates.

        Returns:
            Pixel coordinates, or ``None`` when the point is behind the camera.
        """
        env = self._executor.env
        image_size = self._executor.image_size
        cam_id = env.sim.model.camera_name2id(camera_name)
        fovy = env.sim.model.cam_fovy[cam_id]
        f = 0.5 * image_size.height / np.tan(np.deg2rad(fovy) / 2)
        cx, cy = image_size.width / 2, image_size.height / 2

        x, y, z = xyz_cam
        if -z <= 1e-6:
            return None
        return cx + f * (x / -z), cy - f * (y / -z)
