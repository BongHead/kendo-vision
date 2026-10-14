"""Two fixed player identities on top of the pose tracker, plus red / white assignment."""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from ..config import TrackConfig
from ..constants import LEFT_HIP, LEFT_SHOULDER, MIN_KEYPOINT_CONFIDENCE, RIGHT_HIP, RIGHT_SHOULDER, SIDES
from ..types import Frame, Player
from .constants import TORSO_KEYPOINTS
from .pose import Detection


@dataclass
class _Slot:
    """One of the two players being followed."""
    side: str
    center: np.ndarray
    height: float
    scale: float
    last_seen: float
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))
    track_ids: set = field(default_factory=set)  # pose-tracker ids seen for this player
    red_fraction: float = 0.0  # running share of red pixels on the torso
    votes: int = 0


def _center(detection: Detection) -> np.ndarray:
    left, top, right, bottom = detection.box
    return np.array([(left + right) / 2, (top + bottom) / 2])


def _height(detection: Detection) -> float:
    return float(detection.box[3] - detection.box[1])


def _area(detection: Detection) -> float:
    left, top, right, bottom = detection.box
    return float((right - left) * (bottom - top))


def _torso_length(detection: Detection) -> float | None:
    """Shoulder midpoint to hip midpoint, or None when any of the four keypoints is missing."""
    if detection.keypoint_confidence[TORSO_KEYPOINTS].min() < MIN_KEYPOINT_CONFIDENCE:
        return None
    shoulders = detection.keypoints[[LEFT_SHOULDER, RIGHT_SHOULDER]].mean(axis=0)
    hips = detection.keypoints[[LEFT_HIP, RIGHT_HIP]].mean(axis=0)
    return float(np.linalg.norm(shoulders - hips))


def _scale(detection: Detection) -> float:
    """Torso length, estimated from the box when the torso keypoints are missing."""
    return _torso_length(detection) or 0.3 * _height(detection)


def _two_largest(detections: list[Detection]) -> list[Detection]:
    """The two biggest people, ordered left to right."""
    largest = sorted(detections, key=_area, reverse=True)[:2]
    return sorted(largest, key=lambda detection: _center(detection)[0])


def untracked_players(detections: list[Detection]) -> list[Player]:
    """The two largest detections as players, for single images with no history."""
    return [Player(slot_index, SIDES[slot_index], detection.keypoints, detection.keypoint_confidence,
                   _scale(detection), detection.box)
            for slot_index, detection in enumerate(_two_largest(detections))]


class PlayerTracker:
    """Locks onto the two largest people, then follows them by position and tracker id."""

    def __init__(self, config: TrackConfig):
        self.config = config
        self.slots: list[_Slot] = []

    def update(self, detections: list[Detection], frame: Frame) -> list[Player]:
        config = self.config
        both_lost = self.slots and all(frame.time - slot.last_seen > config.max_lost_seconds for slot in self.slots)
        if both_lost:
            self.slots = []
        if not self.slots:
            if len(detections) < 2:
                return []
            # lock on: the two largest people, the left one in the first slot
            pair = _two_largest(detections)
            sides = SIDES if config.red_side == "left" else SIDES[::-1]
            self.slots = [_Slot(side, _center(detection), _height(detection), _scale(detection), frame.time)
                          for side, detection in zip(sides, pair)]
            matches = dict(enumerate(pair))
        else:
            matches = self._associate(detections, frame.time)
        for slot_index, detection in matches.items():
            self._update_slot(slot_index, detection, frame.time)
        if config.use_tasuki:
            self._vote_tasuki(frame.image, matches)
        return [Player(slot_index, self.slots[slot_index].side, detection.keypoints,
                       detection.keypoint_confidence, self.slots[slot_index].scale, detection.box)
                for slot_index, detection in sorted(matches.items())]

    def _associate(self, detections: list[Detection], now: float) -> dict[int, Detection]:
        """Cheapest assignment of detections to the two slots; a slot may stay unmatched."""
        predicted = [slot.center + slot.velocity * (now - slot.last_seen) for slot in self.slots]
        # the allowed distance grows the longer a slot has been lost
        gates = [self.config.gate * (1 + 2 * (now - slot.last_seen)) for slot in self.slots]

        def cost(slot_index: int, detection: Detection) -> float:
            """Distance from where the slot's player should be, in box heights."""
            slot = self.slots[slot_index]
            distance = float(np.linalg.norm(_center(detection) - predicted[slot_index])) / slot.height
            if detection.track_id >= 0:
                if detection.track_id in slot.track_ids:
                    distance *= 0.5  # the pose tracker agrees this is the same person
                elif detection.track_id in self.slots[1 - slot_index].track_ids:
                    distance += 0.3  # the pose tracker thinks this is the other player
            return distance

        costs = [[cost(slot_index, detection) for detection in detections] for slot_index in range(2)]
        choices = [*range(len(detections)), None]  # None leaves a slot unmatched
        best_matches, best_total = {}, float("inf")
        # With only two slots, simply try every pairing of detections (or nothing) with them.
        for first_choice in choices:
            for second_choice in choices:
                if first_choice is not None and first_choice == second_choice:
                    continue  # one detection cannot be both players
                total, matches = 0.0, {}
                for slot_index, choice in ((0, first_choice), (1, second_choice)):
                    if choice is None:
                        total += gates[slot_index]  # unmatched costs as much as the worst allowed match
                    elif costs[slot_index][choice] > gates[slot_index]:
                        total = float("inf")  # too far away to be this player
                    else:
                        total += costs[slot_index][choice]
                        matches[slot_index] = detections[choice]
                if total < best_total:
                    best_matches, best_total = matches, total
        return best_matches

    def _update_slot(self, slot_index: int, detection: Detection, now: float) -> None:
        slot, center = self.slots[slot_index], _center(detection)
        elapsed = now - slot.last_seen
        if 0 < elapsed < 0.5:
            slot.velocity = 0.5 * slot.velocity + 0.5 * (center - slot.center) / elapsed
        else:  # first sighting, or back after a gap: the old velocity means nothing
            slot.velocity = np.zeros(2)
        slot.center, slot.last_seen = center, now
        slot.height += 0.2 * (_height(detection) - slot.height)
        torso_length = _torso_length(detection)
        if torso_length:
            slot.scale += 0.1 * (torso_length - slot.scale)  # slow, so one bad frame barely moves it
        if detection.track_id >= 0:
            slot.track_ids.add(detection.track_id)
            self.slots[1 - slot_index].track_ids.discard(detection.track_id)

    def _vote_tasuki(self, image: np.ndarray, matches: dict[int, Detection]) -> None:
        """Whichever player shows consistently more red on the torso is red."""
        for slot_index, detection in matches.items():
            if detection.keypoint_confidence[TORSO_KEYPOINTS].min() < MIN_KEYPOINT_CONFIDENCE:
                continue
            slot, torso_points = self.slots[slot_index], detection.keypoints[TORSO_KEYPOINTS]
            padding = 0.25 * slot.scale
            left, top = np.maximum(torso_points.min(axis=0) - padding, 0).astype(int)
            right, bottom = (torso_points.max(axis=0) + padding).astype(int)
            torso_crop = image[top:bottom, left:right]
            if torso_crop.size == 0:
                continue
            hsv = cv2.cvtColor(torso_crop, cv2.COLOR_BGR2HSV)
            # red sits at both ends of the hue circle
            low_reds = cv2.inRange(hsv, (0, 120, 80), (10, 255, 255))
            high_reds = cv2.inRange(hsv, (170, 120, 80), (180, 255, 255))
            red_mask = low_reds | high_reds
            slot.red_fraction += 0.05 * (red_mask.mean() / 255 - slot.red_fraction)
            slot.votes += 1
        first, second = self.slots
        enough_votes = min(first.votes, second.votes) >= 30
        if enough_votes and abs(first.red_fraction - second.red_fraction) > self.config.tasuki_margin:
            if first.red_fraction > second.red_fraction:
                first.side, second.side = "red", "white"
            else:
                first.side, second.side = "white", "red"
