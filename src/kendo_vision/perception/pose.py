"""Body pose for everyone in the frame, via Ultralytics YOLO pose."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import ModelConfig


@dataclass
class Detection:
    box: np.ndarray  # left, top, right, bottom
    score: float
    keypoints: np.ndarray  # (17, 2)
    keypoint_confidence: np.ndarray  # (17,)
    track_id: int = -1  # id from the Ultralytics tracker, -1 if untracked


def predict_args(model_config: ModelConfig, image_size: int, confidence: float) -> dict:
    """Keyword arguments shared by every Ultralytics predict / track call."""
    import torch

    arguments = dict(imgsz=image_size, conf=confidence, verbose=False)
    if model_config.half_precision and torch.cuda.is_available():
        arguments["quantize"] = 16  # FP16
    if model_config.device:
        arguments["device"] = model_config.device
    return arguments


class PoseEstimator:
    def __init__(self, model_config: ModelConfig, tracker: str | None = None):
        """`tracker` is an Ultralytics tracker config; None runs plain per-image prediction."""
        from ultralytics import YOLO

        self.model = YOLO(model_config.pose_weights)
        self.tracker = tracker
        self.arguments = predict_args(model_config, model_config.image_size, model_config.pose_confidence)
        self.arguments["classes"] = [0]  # people only

    def __call__(self, image: np.ndarray) -> list[Detection]:
        if self.tracker:
            result = self.model.track(image, persist=True, tracker=self.tracker, **self.arguments)[0]
        else:
            result = self.model.predict(image, **self.arguments)[0]
        if result.boxes is None or result.keypoints is None or len(result.boxes) == 0:
            return []
        boxes = result.boxes.xyxy.cpu().numpy().astype(float)
        scores = result.boxes.conf.cpu().numpy()
        if result.boxes.id is not None:
            track_ids = result.boxes.id.cpu().numpy().astype(int)
        else:  # the tracker has not confirmed anyone yet
            track_ids = np.full(len(boxes), -1)
        keypoints = result.keypoints.xy.cpu().numpy().astype(float)
        if result.keypoints.conf is not None:
            confidences = result.keypoints.conf.cpu().numpy().astype(float)
        else:
            confidences = np.ones(keypoints.shape[:2])
        return [Detection(box, float(score), points, confidence, int(track_id))
                for box, score, points, confidence, track_id in zip(boxes, scores, keypoints, confidences, track_ids)]
