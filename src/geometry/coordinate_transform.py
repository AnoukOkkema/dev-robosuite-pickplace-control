from typing import Tuple

import numpy as np
from scipy.spatial.transform import Rotation


class CoordinateTransformer:
    """Translate pose-estimator output into positions and orientations for control.

    These static helpers apply the agent-view camera pose and calculate the
    green-axis grasp yaw expected by the controller. They also treat a 180°
    object rotation as equivalent because the supported grasp orientations are
    symmetric around the vertical tool axis.
    """

    @staticmethod
    def camera_to_world_frame(
        xyz_cam: np.ndarray,
        rot_cam: np.ndarray,
        cam_xpos: np.ndarray,
        cam_xmat: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Convert a camera-frame object pose into its world-frame pose.

        The pose estimator labels an object's grasp-alignment direction as
        its green (Y) axis, but the yaw math below treats the grasp axis as
        X. Before extracting yaw, this remaps ``world_xmat``'s columns to
        (Y, -X, Z) so the resulting angle lines up with the controller's
        grasp-alignment convention instead of the estimator's raw object
        frame.

        Args:
            xyz_cam: Camera-frame position (3,).
            rot_cam: Camera-frame rotation matrix (3, 3).
            cam_xpos: Camera position, world-frame (3,).
            cam_xmat: Camera rotation matrix, world-frame (3, 3).

        Returns:
            Tuple[np.ndarray, np.ndarray, float]:
                (world_xpos, world_xmat, green_axis_yaw_deg).
        """

        world_xpos = cam_xmat @ xyz_cam + cam_xpos
        world_xmat = cam_xmat @ rot_cam
        # Remap to the grasp-axis convention: green (Y) axis becomes X.
        green_axis_matrix = np.column_stack(
            (
                world_xmat[:, 1],
                -world_xmat[:, 0],
                world_xmat[:, 2],
            )
        )
        green_axis_yaw_deg = CoordinateTransformer.rotation_matrix_to_yaw_deg(
            green_axis_matrix
        )

        return world_xpos, world_xmat, green_axis_yaw_deg

    @staticmethod
    def quat_to_yaw_deg(quat_xyzw: np.ndarray) -> float:
        """Extract symmetry-wrapped world yaw from an XYZW quaternion.

        Args:
            quat_xyzw: Quaternion (x, y, z, w).

        Returns:
            float: Yaw, degrees.
        """

        rpy = Rotation.from_quat(quat_xyzw).as_euler("xyz", degrees=True)
        return CoordinateTransformer._wrap_yaw(rpy[2])

    @staticmethod
    def rotation_matrix_to_yaw_deg(rotation_matrix: np.ndarray) -> float:
        """Extract symmetry-wrapped world yaw from a rotation matrix.

        Args:
            rotation_matrix: (3, 3) world-frame rotation matrix.

        Returns:
            float: Yaw, degrees.
        """

        rpy = Rotation.from_matrix(rotation_matrix).as_euler("xyz", degrees=True)
        return CoordinateTransformer._wrap_yaw(rpy[2])

    @staticmethod
    def quat_to_roll_pitch_yaw_deg(quat_xyzw: np.ndarray) -> Tuple[float, float, float]:
        """Extract unwrapped roll, pitch, and yaw from an XYZW quaternion.

        Args:
            quat_xyzw: Quaternion (x, y, z, w).

        Returns:
            Tuple[float, float, float]: (roll, pitch, yaw), degrees.
        """

        roll, pitch, yaw = Rotation.from_quat(quat_xyzw).as_euler("xyz", degrees=True)
        return roll, pitch, yaw

    @staticmethod
    def _wrap_yaw(yaw: float) -> float:
        """Wrap yaw into the 180-degree symmetry interval ``[-90, 90]``.

        A 180-degree turn of a symmetric object is an equivalent grasp, so
        folding yaw into a half-turn range keeps the controller from
        commanding a needlessly long rotation to reach an orientation that
        looks identical to one already within reach.

        Args:
            yaw: World yaw in degrees, before symmetry wrapping.
        """
        if yaw > 90:
            yaw -= 180
        elif yaw < -90:
            yaw += 180

        return yaw
