"""Keeps only people who look like kendoka, dropping shinpan and spectators, with zero-shot CLIP."""
from __future__ import annotations

import math

import cv2
import numpy as np

from ..config import TrackConfig
from .constants import CLIP_LOGIT_SCALE, CLIP_MODEL, CLIP_WEIGHTS, MIN_CROP_PIXELS, PERSON_DESCRIPTIONS
from .pose import Detection


class PlayerFilter:
    """Scores each person crop against text descriptions; a tracked person is re-checked now and then."""

    def __init__(self, config: TrackConfig):
        import open_clip
        import torch

        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            CLIP_MODEL, pretrained=CLIP_WEIGHTS, device=self.device)
        self.model.eval()
        tokenizer = open_clip.get_tokenizer(CLIP_MODEL)
        class_embeddings = []
        with torch.no_grad():
            for descriptions in PERSON_DESCRIPTIONS.values():
                embeddings = self.model.encode_text(tokenizer(list(descriptions)).to(self.device))
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
                average = embeddings.mean(dim=0)  # one direction per class, from all its descriptions
                class_embeddings.append(average / average.norm())
        self.class_embeddings = torch.stack(class_embeddings)
        self.track_scores: dict[int, float] = {}  # running kendoka probability per tracker id
        self.last_checked: dict[int, int] = {}  # frame count at which each tracker id was last classified
        self.frame_count = 0

    def __call__(self, image: np.ndarray, detections: list[Detection]) -> list[Detection]:
        self.frame_count += 1
        # untracked people are classified every frame; tracked ones only when their re-check is due
        due = [detection for detection in detections if detection.track_id < 0 or self._recheck_due(detection.track_id)]
        fresh_scores = {}  # by id() of the detection, for this frame only
        for detection, probability in zip(due, self._kendoka_probabilities(image, due)):
            if detection.track_id < 0:
                fresh_scores[id(detection)] = probability
                continue
            previous = self.track_scores.get(detection.track_id)
            # average with earlier checks, so one odd crop (a turn, an overlap) does not flip a person
            self.track_scores[detection.track_id] = probability if previous is None else 0.5 * (previous + probability)
            self.last_checked[detection.track_id] = self.frame_count
        kept = []
        for detection in detections:
            score = fresh_scores.get(id(detection), self.track_scores.get(detection.track_id, 0.0))
            if score >= self.config.player_threshold:
                kept.append(detection)
        return kept

    def _recheck_due(self, track_id: int) -> bool:
        frames_since = self.frame_count - self.last_checked.get(track_id, -math.inf)
        return frames_since >= self.config.reclassify_every

    def _kendoka_probabilities(self, image: np.ndarray, detections: list[Detection]) -> list[float]:
        """Probability that each detection is a kendoka, classified together in one batch."""
        import torch
        from PIL import Image

        if not detections:
            return []
        height, width = image.shape[:2]
        crops, valid = [], []
        for detection in detections:
            left, top, right, bottom = detection.box.astype(int)
            left, top, right, bottom = max(left, 0), max(top, 0), min(right, width), min(bottom, height)
            if right - left < MIN_CROP_PIXELS or bottom - top < MIN_CROP_PIXELS:
                valid.append(False)  # too small to judge
                continue
            crop = cv2.cvtColor(image[top:bottom, left:right], cv2.COLOR_BGR2RGB)
            crops.append(self.preprocess(Image.fromarray(crop)))
            valid.append(True)
        probabilities = []
        if crops:
            with torch.no_grad():
                features = self.model.encode_image(torch.stack(crops).to(self.device))
                features = features / features.norm(dim=-1, keepdim=True)
                class_probabilities = (CLIP_LOGIT_SCALE * features @ self.class_embeddings.T).softmax(dim=-1)
                probabilities = class_probabilities[:, 0].float().cpu().tolist()  # column 0 is "kendoka"
        # put the scores back in detection order, with 0 for crops that were too small
        scores = iter(probabilities)
        return [next(scores) if is_valid else 0.0 for is_valid in valid]
