"""Estimate each detected object's 3D camera-frame position and orientation."""

import logging
from typing import Tuple

import cv2
import numpy as np
import onnxruntime as ort


class PoseEstimator:
    """Estimate the 3D pose of one object detected in the agent-view image.

    The model receives the full camera image, the detection box, the detected
    class, and an object crop. It returns XYZ relative to the camera plus a 6D
    rotation representation. The executor later converts this output to world
    coordinates before sending it to the robot controller.

    Attributes:
        num_classes: Number of classes accepted by the model.
        pos_image_size: Full-image input size for position inference.
        rotation_image_size: Object-crop input size for rotation inference.
        session: Configured ONNX Runtime inference session.
        input_names: Ordered names of the ONNX input tensors.
        logger: Logger used for initialization details.
    """

    def __init__(
        self,
        model_path: str,
        num_classes: int,
        pos_image_size: int = 224,
        rotation_image_size: int = 128,
        intra_op_threads: int = 4,
        inter_op_threads: int = 4,
        logger=None,
    ) -> None:
        """Load the ONNX pose-estimation model.

        Args:
            model_path: Path to the pose-estimation ONNX model.
            num_classes: Number of detector classes used by the model.
            pos_image_size: Full-image input size for position inference.
            rotation_image_size: Object-crop input size for rotation inference.
            intra_op_threads: Threads used inside individual inference operators.
            inter_op_threads: Threads used between inference operators.
            logger: Logger used for initialization details.
        """
        self.logger = logger or logging.getLogger(__name__)

        self.num_classes = num_classes
        self.pos_image_size = pos_image_size
        self.rotation_image_size = rotation_image_size

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = max(1, int(intra_op_threads))
        session_options.inter_op_num_threads = max(1, int(inter_op_threads))

        self.session = ort.InferenceSession(
            model_path,
            providers=providers,
            sess_options=session_options,
        )

        self.input_names = [i.name for i in self.session.get_inputs()]

        self.logger.info(
            "PoseEstimator initialized | model=%s | inputs=%s | providers=%s | "
            "active=%s",
            model_path,
            self.input_names,
            providers,
            self.session.get_providers(),
        )

    @staticmethod
    def rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
        """Reconstruct a 3-by-3 rotation matrix from a 6D representation."""

        # Gram-Schmidt keeps the model's two predicted axes orthonormal.
        a1, a2 = rot6d[:3], rot6d[3:]

        b1 = a1 / np.linalg.norm(a1)
        b2 = a2 - np.dot(b1, a2) * b1
        b2 = b2 / np.linalg.norm(b2)
        b3 = np.cross(b1, b2)

        return np.column_stack([b1, b2, b3])

    def _preprocess_image(self, image_bgr: np.ndarray, image_size: int) -> np.ndarray:
        """Resize a BGR image and convert it into a normalized NCHW tensor."""
        resized = cv2.resize(image_bgr, (image_size, image_size))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        chw = np.transpose(rgb, (2, 0, 1))
        return np.expand_dims(chw, axis=0)

    def _build_bbox_features(
        self, bbox, frame_width: int, frame_height: int
    ) -> np.ndarray:
        """Build normalized bounding-box features expected by the ONNX model."""
        x1, y1, x2, y2 = bbox
        x1n, y1n, x2n, y2n = (
            x1 / frame_width,
            y1 / frame_height,
            x2 / frame_width,
            y2 / frame_height,
        )
        area = (x2n - x1n) * (y2n - y1n)
        cx, cy = (x1n + x2n) / 2, (y1n + y2n) / 2
        return np.array([[x1n, y1n, x2n, y2n, area, cx, cy]], dtype=np.float32)

    def _build_class_onehot(self, class_id: int) -> np.ndarray:
        """Build the one-hot encoded class input expected by the ONNX model."""
        onehot = np.zeros((1, self.num_classes), dtype=np.float32)
        onehot[0, class_id] = 1.0
        return onehot

    def predict(
        self,
        image_bgr: np.ndarray,
        bbox: Tuple[float, float, float, float],
        class_id: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict the camera-frame position and rotation of one detection.

        Args:
            image_bgr (np.ndarray): Full agentview frame (H, W, 3), BGR.
            bbox (Tuple[float, float, float, float]):
                (x1, y1, x2, y2) in ``image_bgr`` pixel space.
            class_id (int): Class index matching training's class_names order.

        Returns:
            Camera-frame position and a 3-by-3 rotation matrix.
        """

        frame_height, frame_width = image_bgr.shape[:2]

        image_blob = self._preprocess_image(image_bgr, self.pos_image_size)
        bbox_features = self._build_bbox_features(bbox, frame_width, frame_height)
        class_onehot = self._build_class_onehot(class_id)

        x1, y1, x2, y2 = (int(round(v)) for v in bbox)
        x1c, y1c = max(x1, 0), max(y1, 0)
        x2c, y2c = min(x2, frame_width), min(y2, frame_height)
        object_crop = image_bgr[y1c:y2c, x1c:x2c]

        if object_crop.size == 0:
            # A clipped or invalid detection still receives a valid model input.
            object_crop = image_bgr

        crop_blob = self._preprocess_image(object_crop, self.rotation_image_size)

        xyz_pred, rot6d_pred = self.session.run(
            None,
            {
                # Input order follows the pose-estimator ONNX export contract.
                self.input_names[0]: image_blob,
                self.input_names[1]: bbox_features,
                self.input_names[2]: class_onehot,
                self.input_names[3]: crop_blob,
            },
        )

        xyz_cam = xyz_pred[0]
        rot_cam = self.rot6d_to_matrix(rot6d_pred[0])

        return xyz_cam, rot_cam
