import logging
from typing import List

import cv2
import numpy as np
import onnxruntime as ort

from src.util.types import Detection


class OnnxDetector:
    """Detect Bread, Can, Cereal, and Milk in one cropped agent-view image.

    Images are letterboxed: resized to fit inside a square canvas with
    aspect ratio preserved, then black-padded, rather than squashed to a
    square. This exactly mirrors the "Fit (black edges)" preprocessing
    Roboflow applied when building the training dataset in the vision repo;
    squashing here would show the model a geometry it never trained on.
    Post-processing then maps predicted boxes back out of letterboxed
    coordinates into the original image, so they line up with the image
    passed to the pose-estimation model rather than the padded square only
    the detector saw.
    """

    LETTERBOX_PAD_COLOR = (0, 0, 0)

    def __init__(
        self,
        model_path: str,
        class_names: List[str],
        image_size: int = 640,
        intra_op_threads: int = 4,
        inter_op_threads: int = 4,
        logger=None,
    ) -> None:
        """Load the ONNX detector.

        Args:
            model_path: Path to the YOLO ONNX model.
            class_names: Class names indexed by model output class ID.
            image_size: Square model input dimension in pixels.
            intra_op_threads: Threads used inside individual inference operators.
            inter_op_threads: Threads used between inference operators.
            logger: Logger used for initialization details.
        """
        self.logger = logger or logging.getLogger(__name__)

        self.class_names = class_names
        self.image_size = image_size

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = max(1, int(intra_op_threads))
        session_options.inter_op_num_threads = max(1, int(inter_op_threads))

        self.session = ort.InferenceSession(
            model_path,
            providers=providers,
            sess_options=session_options,
        )

        self.input_name = self.session.get_inputs()[0].name

        self.logger.info(
            "OnnxDetector initialized | model=%s | classes=%s | providers=%s | "
            "active=%s",
            model_path,
            self.class_names,
            providers,
            self.session.get_providers(),
        )

    def _letterbox(self, image: np.ndarray):
        """Resize ``image`` into the square model input with black padding.

        Args:
            image: Source image to letterbox into the square model input.
        """
        height, width = image.shape[:2]

        scale = min(self.image_size / height, self.image_size / width)

        new_height = round(height * scale)
        new_width = round(width * scale)

        resized = cv2.resize(
            image, (new_width, new_height), interpolation=cv2.INTER_LINEAR
        )

        canvas = np.full(
            (self.image_size, self.image_size, 3),
            self.LETTERBOX_PAD_COLOR,
            dtype=np.uint8,
        )

        pad_top = (self.image_size - new_height) // 2
        pad_left = (self.image_size - new_width) // 2

        # Keep padding offsets so predicted boxes can be mapped back exactly.
        canvas[pad_top : pad_top + new_height, pad_left : pad_left + new_width] = (
            resized
        )

        return canvas, scale, pad_left, pad_top

    def _to_blob(self, letterboxed_bgr: np.ndarray) -> np.ndarray:
        """Convert a BGR image into a normalized NCHW inference tensor.

        Args:
            letterboxed_bgr: Square, letterboxed BGR image produced by
                ``_letterbox``, to convert into the model's input tensor.
        """
        rgb = cv2.cvtColor(letterboxed_bgr, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        chw = np.transpose(normalized, (2, 0, 1))
        return np.expand_dims(chw, axis=0)

    def _postprocess(
        self,
        output,
        scale,
        pad_left,
        pad_top,
        original_size,
        conf_threshold,
        iou_threshold,
    ) -> List[Detection]:
        """Convert raw YOLO predictions to original-image detections.

        Args:
            output: Raw model output for one image, with anchors as columns
                and (4 box coords + per-class scores) as rows.
            scale: Letterbox scale factor to divide out when mapping boxes
                back to the original image.
            pad_left: Horizontal letterbox padding to subtract when mapping
                boxes back to the original image.
            pad_top: Vertical letterbox padding to subtract when mapping
                boxes back to the original image.
            original_size: (height, width) of the original, un-letterboxed
                image, used to clip mapped boxes to valid bounds.
            conf_threshold: Minimum class confidence a box must have to be
                kept before non-max suppression.
            iou_threshold: IoU threshold passed to non-max suppression for
                discarding overlapping boxes.
        """
        # YOLO exports anchors as columns. OpenCV expects rows for NMS.
        predictions = output.T  # (num_anchors, 4 + nc)

        boxes_cxcywh = predictions[:, :4]
        class_scores = predictions[:, 4:]

        class_ids = np.argmax(class_scores, axis=1)
        confidences = class_scores[np.arange(len(class_scores)), class_ids]

        keep_mask = confidences >= conf_threshold

        boxes_cxcywh = boxes_cxcywh[keep_mask]
        confidences = confidences[keep_mask]
        class_ids = class_ids[keep_mask]

        if len(boxes_cxcywh) == 0:
            return []

        boxes_xywh = np.column_stack(
            [
                boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2,
                boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2,
                boxes_cxcywh[:, 2],
                boxes_cxcywh[:, 3],
            ]
        )

        keep_indices = cv2.dnn.NMSBoxes(
            boxes_xywh.tolist(),
            confidences.tolist(),
            conf_threshold,
            iou_threshold,
        )

        original_height, original_width = original_size

        detections = []

        for index in np.array(keep_indices).flatten():
            x, y, w, h = boxes_xywh[index]

            x1 = (x - pad_left) / scale
            y1 = (y - pad_top) / scale
            x2 = (x + w - pad_left) / scale
            y2 = (y + h - pad_top) / scale

            detections.append(
                Detection(
                    box=(
                        float(np.clip(x1, 0, original_width)),
                        float(np.clip(y1, 0, original_height)),
                        float(np.clip(x2, 0, original_width)),
                        float(np.clip(y2, 0, original_height)),
                    ),
                    confidence=float(confidences[index]),
                    class_id=int(class_ids[index]),
                )
            )

        return detections

    def predict(
        self,
        image: np.ndarray,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> List[Detection]:
        """Run detection on one BGR image.

        Args:
            image (np.ndarray): Source image (H, W, 3) in BGR.
            conf_threshold (float): Minimum class confidence to keep a box.
            iou_threshold (float): IoU threshold used for NMS.

        Returns:
            Detections in the original image coordinate space.
        """

        letterboxed, scale, pad_left, pad_top = self._letterbox(image)
        blob = self._to_blob(letterboxed)

        out = self.session.run(None, {self.input_name: blob})[0][0]

        return self._postprocess(
            output=out,
            scale=scale,
            pad_left=pad_left,
            pad_top=pad_top,
            original_size=image.shape[:2],
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
        )
