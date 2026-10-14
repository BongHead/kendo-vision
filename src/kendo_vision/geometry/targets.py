"""Target zones (datotsu-bui) built from body keypoints, and shinai-to-zone distances."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import ZoneConfig
from ..constants import (FACE_KEYPOINTS, LEFT_ELBOW, LEFT_HIP, LEFT_SHOULDER, LEFT_WRIST, RIGHT_ELBOW, RIGHT_HIP,
                         RIGHT_SHOULDER, RIGHT_WRIST, TARGETS)
from ..types import Player, Shinai


@dataclass
class Zone:
    """A capsule: every point within `radius` of the segment start-end. A circle when they coincide."""
    start: np.ndarray
    end: np.ndarray
    radius: float


def target_zones(player: Player, zone_config: ZoneConfig) -> dict[str, list[Zone]]:
    """Zones per target for one player; a target with missing keypoints gets no zone."""
    scale = player.scale
    zones: dict[str, list[Zone]] = {target: [] for target in TARGETS}
    shoulders = player.shoulders
    face = [player.keypoints[index] for index in FACE_KEYPOINTS if player.has(index)]
    if face:
        head = np.mean(face, axis=0)
    elif shoulders is not None:
        # no face keypoints (common under a men): place the head above the shoulders, along the spine
        spine = shoulders - player.hips
        head = shoulders + 0.45 * scale * spine / (np.linalg.norm(spine) + 1e-9)
    else:
        head = None
    if head is not None:
        zones["men"].append(Zone(head, head, zone_config.men_radius * scale))
        if shoulders is not None:
            throat = shoulders + 0.35 * (head - shoulders)
            zones["tsuki"].append(Zone(throat, throat, zone_config.tsuki_radius * scale))

    # Right kote is always a target; left kote only while that hand is raised above the shoulders.
    for wrist, elbow, always_valid in ((RIGHT_WRIST, RIGHT_ELBOW, True), (LEFT_WRIST, LEFT_ELBOW, False)):
        if not player.has(wrist, elbow):
            continue
        wrist_point = player.keypoints[wrist]
        hand_raised = shoulders is not None and wrist_point[1] <= shoulders[1]  # image y grows downward
        if not always_valid and not hand_raised:
            continue
        toward_elbow = player.keypoints[elbow] - wrist_point
        zones["kote"].append(Zone(wrist_point, wrist_point + zone_config.kote_length * toward_elbow,
                                  zone_config.kote_radius * scale))

    # One do zone down each side of the torso, over its lower part.
    for hip, shoulder in ((RIGHT_HIP, RIGHT_SHOULDER), (LEFT_HIP, LEFT_SHOULDER)):
        if player.has(hip, shoulder):
            hip_point = player.keypoints[hip]
            up_the_side = player.keypoints[shoulder] - hip_point
            zones["do"].append(Zone(hip_point + zone_config.do_low * up_the_side,
                                    hip_point + zone_config.do_high * up_the_side,
                                    zone_config.do_radius * scale))
    return zones


def _cross(first: np.ndarray, second: np.ndarray) -> float:
    """2D cross product: positive when `second` is anticlockwise from `first`."""
    return float(first[0] * second[1] - first[1] * second[0])


def _closest_on_segment(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> tuple[float, float]:
    """Distance from `point` to the segment, and how far along it (0..1) the closest point lies."""
    along = end - start
    length_squared = float(along @ along)
    if length_squared < 1e-12:  # the segment is a single point
        position = 0.0
    else:
        position = float(np.clip((point - start) @ along / length_squared, 0.0, 1.0))
    return float(np.linalg.norm(point - (start + position * along))), position


def segment_distance(first_start: np.ndarray, first_end: np.ndarray,
                     second_start: np.ndarray, second_end: np.ndarray) -> tuple[float, float]:
    """Distance between two 2D segments, and how far along the first (0..1) its closest point lies."""
    first_along, second_along = first_end - first_start, second_end - second_start
    denominator = _cross(first_along, second_along)
    if abs(denominator) > 1e-12:  # not parallel, so the two lines meet somewhere
        between_starts = second_start - first_start
        first_position = _cross(between_starts, second_along) / denominator
        second_position = _cross(between_starts, first_along) / denominator
        if 0 <= first_position <= 1 and 0 <= second_position <= 1:
            return 0.0, first_position  # they meet within both segments: the segments cross
    # Otherwise the closest approach involves an endpoint of one of the segments.
    best = (float("inf"), 0.0)
    for endpoint, position in ((first_start, 0.0), (first_end, 1.0)):
        distance, _ = _closest_on_segment(endpoint, second_start, second_end)
        best = min(best, (distance, position))
    for endpoint in (second_start, second_end):
        best = min(best, _closest_on_segment(endpoint, first_start, first_end))
    return best


def zone_gap(start: np.ndarray, end: np.ndarray, zone: Zone) -> float:
    """Gap between the segment start-end and the zone surface (0 when touching)."""
    return max(0.0, segment_distance(start, end, zone.start, zone.end)[0] - zone.radius)


def strike_gaps(shinai: Shinai, zones: dict[str, list[Zone]], scale: float,
                monouchi_fraction: float) -> dict[str, tuple[float, float]]:
    """Per target: (gap from the striking part to the zone in torso lengths,
    position along the blade of the point nearest the zone, 0 = tip, 1 = tsuba)."""
    tip, tsuba = shinai.tip, shinai.tsuba
    monouchi_end = tip + monouchi_fraction * (tsuba - tip)
    gaps = {}
    for target, target_zones_list in zones.items():
        nearest = None
        for zone in target_zones_list:
            # tsuki is a thrust, so only the tip counts; every other target is struck with the monouchi
            striking_end = tip if target == "tsuki" else monouchi_end
            gap = zone_gap(tip, striking_end, zone) / scale
            if nearest is None or gap < nearest[0]:
                blade_position = segment_distance(tip, tsuba, zone.start, zone.end)[1]
                nearest = (gap, blade_position)
        if nearest is not None:
            gaps[target] = nearest
    return gaps
