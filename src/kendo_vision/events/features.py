"""Per-frame observations, and the player-relative feature windows built from them."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from ..config import Config
from ..constants import KEYPOINT_COUNT, MIN_KEYPOINT_CONFIDENCE, SIDES, TARGETS
from ..geometry.targets import Zone, strike_gaps, target_zones
from ..types import Player, Shinai
from .constants import (AUDIO_BAND_COUNT, FEATURE_GAPS_START, FEATURE_KEYPOINT_CONFIDENCE, FEATURE_KEYPOINTS,
                        FEATURE_PRESENT, FEATURE_SHINAI, FEATURE_SHINAI_CONFIDENCE, FEATURES_PER_SIDE, MAX_GAP,
                        RAW_KEYPOINT_CONFIDENCE, RAW_KEYPOINTS, RAW_PER_SIDE, RAW_PRESENT, RAW_SCALE, RAW_SHINAI,
                        RAW_SHINAI_CONFIDENCE, RAW_SIZE, STATIC_SIZE)


@dataclass
class Observation:
    """Everything known about one frame."""
    time: float
    frame_index: int
    players: dict[str, Player]  # by side
    zones: dict[str, dict[str, list[Zone]]]  # side -> target -> zones


class FeatureBuffer:
    """The last few seconds of observations."""

    def __init__(self, seconds: float):
        self.seconds = seconds
        self.observations: deque[Observation] = deque()

    def push(self, observation: Observation) -> None:
        self.observations.append(observation)
        while self.observations[0].time < observation.time - self.seconds:
            self.observations.popleft()

    def span(self, start: float, end: float) -> list[Observation]:
        return [observation for observation in self.observations if start <= observation.time <= end]

    def raw(self) -> tuple[np.ndarray, np.ndarray]:
        """(times, raw rows) for everything in the buffer."""
        times = np.array([observation.time for observation in self.observations])
        return times, np.stack([raw_row(observation.players) for observation in self.observations])


def raw_row(players: dict[str, Player]) -> np.ndarray:
    """Everything the models need from one frame, as a flat vector (layout in constants.py)."""
    row = np.zeros(RAW_SIZE, np.float32)
    for side_index, side in enumerate(SIDES):
        player = players.get(side)
        if player is None:
            continue  # the block stays zero, with its `present` flag off
        side_row = row[side_index * RAW_PER_SIDE:(side_index + 1) * RAW_PER_SIDE]  # a view into `row`
        side_row[RAW_PRESENT], side_row[RAW_SCALE] = 1.0, player.scale
        side_row[RAW_KEYPOINTS] = player.keypoints.reshape(-1)
        side_row[RAW_KEYPOINT_CONFIDENCE] = player.keypoint_confidence
        if player.shinai is not None:
            side_row[RAW_SHINAI] = np.concatenate([player.shinai.butt, player.shinai.tsuba, player.shinai.tip])
            side_row[RAW_SHINAI_CONFIDENCE] = player.shinai.confidence
    return row


def _player_from_row(side_row: np.ndarray, slot: int, side: str) -> Player | None:
    """Rebuild a player from one side's block of a raw row."""
    if side_row[RAW_PRESENT] < 0.5:
        return None
    side_row = side_row.astype(float)
    shinai = None
    if side_row[RAW_SHINAI_CONFIDENCE] > 0:
        butt, tsuba, tip = side_row[RAW_SHINAI].reshape(3, 2)
        shinai = Shinai(butt, tsuba, tip, side_row[RAW_SHINAI_CONFIDENCE])
    return Player(slot, side, side_row[RAW_KEYPOINTS].reshape(KEYPOINT_COUNT, 2),
                  side_row[RAW_KEYPOINT_CONFIDENCE], side_row[RAW_SCALE], shinai=shinai)


def static_features(raw_rows: np.ndarray, config: Config) -> np.ndarray:
    """(frames, RAW_SIZE) raw rows -> (frames, STATIC_SIZE), centred between the players and in torso lengths."""
    features = np.zeros((len(raw_rows), STATIC_SIZE), np.float32)
    features[:, FEATURE_GAPS_START:] = MAX_GAP
    for row_index, row in enumerate(raw_rows):
        players = [_player_from_row(row[side_index * RAW_PER_SIDE:(side_index + 1) * RAW_PER_SIDE], side_index, side)
                   for side_index, side in enumerate(SIDES)]
        present = [player for player in players if player is not None]
        if not present:
            continue
        # Measuring from the midpoint between the players, in torso lengths, cancels camera pans and zooms.
        origin = np.mean([player.hips for player in present], axis=0)
        scale = float(np.mean([player.scale for player in present]))
        zones = [target_zones(player, config.zones) if player is not None else None for player in players]
        for side_index, player in enumerate(players):
            if player is None:
                continue
            side_features = features[row_index, side_index * FEATURES_PER_SIDE:(side_index + 1) * FEATURES_PER_SIDE]
            visible = (player.keypoint_confidence >= MIN_KEYPOINT_CONFIDENCE)[:, None]
            side_features[FEATURE_PRESENT] = 1.0
            # multiplying by `visible` leaves missing keypoints at zero instead of at a wild guess
            side_features[FEATURE_KEYPOINTS] = ((player.keypoints - origin) / scale * visible).reshape(-1)
            side_features[FEATURE_KEYPOINT_CONFIDENCE] = player.keypoint_confidence
            if player.shinai is None:
                continue
            shinai_points = np.stack([player.shinai.butt, player.shinai.tsuba, player.shinai.tip])
            side_features[FEATURE_SHINAI] = ((shinai_points - origin) / scale).reshape(-1)
            side_features[FEATURE_SHINAI_CONFIDENCE] = player.shinai.confidence
            opponent = players[1 - side_index]
            if opponent is None:
                continue
            # how close this player's shinai is to each of the opponent's targets
            gaps = strike_gaps(player.shinai, zones[1 - side_index], opponent.scale, config.shinai.monouchi_fraction)
            first_gap_column = FEATURE_GAPS_START + side_index * len(TARGETS)
            for target_index, target in enumerate(TARGETS):
                if target in gaps:
                    features[row_index, first_gap_column + target_index] = min(gaps[target][0], MAX_GAP)
    return features


def resample(times: np.ndarray, values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Rows of `values` linearly interpolated onto the times in `grid`; the ends are held."""
    if len(times) < 2:
        return np.repeat(values[:1], len(grid), axis=0)
    # for each grid time, the samples just before and just after it
    after = np.clip(np.searchsorted(times, grid), 1, len(times) - 1)
    before = after - 1
    gap = np.maximum(times[after] - times[before], 1e-9)
    weight = np.clip((grid - times[before]) / gap, 0, 1)[:, None]
    return values[before] * (1 - weight) + values[after] * weight


def build_window(times: np.ndarray, static: np.ndarray, center_time: float, config: Config,
                 envelopes=None, augment=None) -> np.ndarray:
    """(steps, FEATURE_SIZE) network input around a strike time, at a fixed rate whatever the video fps."""
    net = config.net
    steps_before, steps_after = int(net.before_seconds * net.rate_hz), int(net.after_seconds * net.rate_hz)
    grid = center_time + np.arange(-steps_before, steps_after) / net.rate_hz
    window = resample(times, static, grid).astype(np.float32)
    if augment is not None:
        window = augment(window)  # before the velocities, so they stay consistent with the positions
    velocity = np.diff(window, axis=0, prepend=window[:1]) * (net.rate_hz / 10)
    audio = np.zeros((len(grid), AUDIO_BAND_COUNT), np.float32)
    if envelopes is not None:
        levels = resample(envelopes[0], envelopes[1], grid)
        # relative to the window's own typical level, so recording volume does not matter
        audio = ((levels - np.median(levels, axis=0)) / 20).astype(np.float32)
    return np.concatenate([window, velocity, audio], axis=1)
