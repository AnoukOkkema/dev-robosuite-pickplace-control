import logging
import os

import cv2
import numpy as np

from src.control.stage import Stage
from src.recording.run_observer import BaseRunObserver
from src.util.types import RunResult


class ScanImageObserver(BaseRunObserver):
    """Save one annotated agent-view PNG per scan, plus a raw opening capture.

    The raw, unannotated first-scan frame is kept for documentation and the
    portfolio case study. Every scan after that gets its own annotated PNG
    with a box, label, and pose axes per detected object.
    """

    def __init__(self, output_dir: str, logger=None) -> None:
        """Create the observer.

        Args:
            output_dir: Directory the scan PNGs are written into.
            logger: Logger used to report each saved file.
        """
        self._output_dir = output_dir
        self._logger = logger or logging.getLogger(__name__)
        self._executor = None
        self._scan_index = 0
        self.axis_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # X, Y, Z in BGR

    def on_run_start(self, executor) -> None:
        """Resolve the executor and reset the per-run scan counter.

        Args:
            executor: The executor running the pick-and-place episode, used
                to look up the crop region, image size, and camera intrinsics.
        """
        self._executor = executor
        self._scan_index = 0
        os.makedirs(self._output_dir, exist_ok=True)

    def on_scan(self, agentview_frame: np.ndarray, poses_cam: list) -> None:
        """Save the raw first scan once, then an annotated image per scan.

        Args:
            agentview_frame: Full, uncropped BGR agent-view capture the scan
                ran detection on; saved as-is for the first scan.
            poses_cam: ``(class_name, box, xyz_cam, rot_cam)`` per detected
                target-class object, drawn onto the annotated copy.
        """
        self._scan_index += 1
        if self._scan_index == 1:
            cv2.imwrite(f"{self._output_dir}/raw_agentview.png", agentview_frame)

        annotated = agentview_frame.copy()
        for class_name, box, xyz_cam, rot_cam in poses_cam:
            annotated = self._annotate(annotated, class_name, box, xyz_cam, rot_cam)

        output_path = f"{self._output_dir}/scan_{self._scan_index:02d}_agentview.png"
        cv2.imwrite(output_path, annotated)
        self._logger.info("Scan %d | saved %s", self._scan_index, output_path)

    def on_target_detected(self, class_name: str, pose_cam: tuple) -> None:
        """Unused. Scan images are built from ``on_scan`` instead.

        Args:
            class_name: Detected object class.
            pose_cam: ``(class_name, box, xyz_cam, rot_cam)`` detection box
                in cropped agent-view pixels plus the camera-frame pose.
        """

    def on_step(self, object_name: str, stage: Stage) -> None:
        """Unused. This observer only reacts to scans.

        Args:
            object_name: Object the controller is currently handling.
            stage: Current pick-and-place stage.
        """

    def on_run_end(self, result: RunResult) -> None:
        """Unused. Nothing to close.

        Args:
            result: Placement/vision-accuracy summary for the finished run.
                Unused; this observer has nothing to close.
        """
        del result

    def _annotate(
        self,
        frame: np.ndarray,
        class_name: str,
        box: tuple,
        xyz_cam: np.ndarray,
        rot_cam: np.ndarray,
    ) -> np.ndarray:
        """Draw one detection's full-frame box, label, and pose axes.

        Args:
            frame: Agent-view frame to draw the annotation onto.
            class_name: Detected object class, drawn as the box label.
            box: Detection box in cropped agent-view pixels; offset by the
                crop region before drawing.
            xyz_cam: Camera-frame position of the detected object.
            rot_cam: Camera-frame rotation matrix of the detected object.
        """
        crop_region = self._executor.crop_region
        x1, y1, x2, y2 = (int(round(value)) for value in box)
        full_frame_box = (
            x1 + crop_region.x1,
            y1 + crop_region.y1,
            x2 + crop_region.x1,
            y2 + crop_region.y1,
        )
        annotated = self._draw_pose_axes(frame, full_frame_box, xyz_cam, rot_cam)
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

    def _draw_pose_axes(
        self,
        frame: np.ndarray,
        box: tuple,
        xyz_cam: np.ndarray,
        rot_cam: np.ndarray,
        axis_length: float = 0.05,
    ) -> np.ndarray:
        """Draw a bounding box and red-X, green-Y, blue-Z pose axes.

        Args:
            frame: Frame to draw the box and axes onto; not modified in place.
            box: Full-frame bounding box to draw.
            xyz_cam: Camera-frame position of the axes' origin.
            rot_cam: Camera-frame rotation matrix defining the axes' directions.
            axis_length: Length of each drawn axis, in metres.
        """
        annotated = frame.copy()
        x1, y1, x2, y2 = box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 1)

        origin = self._project_point(xyz_cam)
        if origin is None:
            return annotated

        ox, oy = origin
        cv2.circle(annotated, (int(ox), int(oy)), 4, (255, 255, 255), -1)
        for axis_index, color in enumerate(self.axis_colors):
            axis_point_cam = xyz_cam + rot_cam[:, axis_index] * axis_length
            axis_pixel = self._project_point(axis_point_cam)
            if axis_pixel is None:
                continue
            ax, ay = axis_pixel
            cv2.line(
                annotated, (int(ox), int(oy)), (int(ax), int(ay)), color, 2, cv2.LINE_AA
            )
        return annotated

    def _project_point(self, xyz_cam: np.ndarray):
        """Project a camera-frame point into full-frame agent-view pixel coordinates.

        Args:
            xyz_cam: Point position in the agent-view camera's frame.

        Returns:
            Pixel coordinates, or ``None`` when the point is behind the camera.
        """
        env = self._executor.env
        image_size = self._executor.image_size
        cam_id = env.sim.model.camera_name2id("agentview")
        fovy = env.sim.model.cam_fovy[cam_id]
        f = 0.5 * image_size.height / np.tan(np.deg2rad(fovy) / 2)
        cx, cy = image_size.width / 2, image_size.height / 2

        x, y, z = xyz_cam
        if -z <= 1e-6:
            return None
        return cx + f * (x / -z), cy - f * (y / -z)
