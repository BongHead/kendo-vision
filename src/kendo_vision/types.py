"""Shared data types."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .constants import (KEYPOINT_COUNT, LEFT_HIP, LEFT_SHOULDER, MIN_KEYPOINT_CONFIDENCE, RIGHT_HIP,
                        RIGHT_SHOULDER)


def other(side: str) -> str:
    return "white" if side == "red" else "red"


@dataclass
class Frame:
    index: int
    time: float  # seconds since the source started
    image: np.ndarray  # BGR


@dataclass
class Shinai:
    butt: np.ndarray
    tsuba: np.ndarray
    tip: np.ndarray
    confidence: float
    source: str = "model"  # which estimator produced it: model, yoloe or wrist


@dataclass
class Player:
    slot: int
    side: str
    keypoints: np.ndarray  # (17, 2) pixels
    keypoint_confidence: np.ndarray  # (17,)
    scale: float  # smoothed torso length in pixels
    box: np.ndarray | None = None  # left, top, right, bottom
    shinai: Shinai | None = None

    def has(self, *indices: int) -> bool:
        """True when every listed keypoint was detected confidently."""
        return all(self.keypoint_confidence[index] >= MIN_KEYPOINT_CONFIDENCE for index in indices)

    def _mean_of(self, indices) -> np.ndarray | None:
        visible = [self.keypoints[index] for index in indices if self.has(index)]
        return np.mean(visible, axis=0) if visible else None

    @property
    def shoulders(self) -> np.ndarray | None:
        return self._mean_of((LEFT_SHOULDER, RIGHT_SHOULDER))

    @property
    def hips(self) -> np.ndarray:
        """Hip midpoint, falling back to the keypoint or box centre."""
        center = self._mean_of((LEFT_HIP, RIGHT_HIP))
        if center is None:
            center = self._mean_of(range(KEYPOINT_COUNT))
        if center is None and self.box is not None:
            left, top, right, bottom = self.box
            center = np.array([(left + right) / 2, (top + bottom) / 2])
        return center if center is not None else np.zeros(2)


def _rounded(value) -> float | None:
    return None if value is None else round(float(value), 3)


@dataclass
class StrikeEvent:
    time: float
    frame: int
    attacker: str
    target: str
    target_confidence: float
    criteria: dict = field(default_factory=dict)  # criterion -> 0..1, or None if unobservable
    validity: float | None = None
    verdict: str | None = None  # ippon | borderline | no
    counted: bool = False  # added to the score

    def to_dict(self) -> dict:
        return {
            "time": _rounded(self.time), "frame": self.frame, "attacker": self.attacker,
            "target": self.target, "target_confidence": _rounded(self.target_confidence),
            "criteria": {name: _rounded(score) for name, score in self.criteria.items()},
            "validity": _rounded(self.validity), "verdict": self.verdict, "counted": self.counted,
        }
