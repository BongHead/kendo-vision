"""Rule-based strike detector: a fast monouchi arriving on a target zone."""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from ..config import Config
from ..constants import SIDES
from ..geometry.targets import strike_gaps, zone_gap
from ..types import other
from .constants import HISTORY_LENGTH, MIN_TIP_SPEED_SPAN
from .features import Observation


@dataclass
class Candidate:
    time: float
    frame: int
    attacker: str
    target: str
    target_confidence: float
    gap: float  # distance left to the zone at contact, torso lengths
    blade_position: float  # contact position along the blade: 0 = tip, 1 = tsuba
    peak_speed: float  # tip speed into contact, torso lengths per second


@dataclass
class _Measure:
    """One attacker's shinai against the defender's targets in one frame."""
    time: float
    frame: int
    gaps: dict[str, tuple[float, float]]  # target -> (gap, blade position)
    tip: np.ndarray  # relative to the defender's hips, torso lengths
    source: str  # the shinai estimator that produced this frame's shinai


class StrikeDetector:
    """Fires when the gap to a zone bottoms out under the contact threshold after a fast approach."""

    def __init__(self, config: Config):
        self.config = config
        self.history: dict[str, deque[_Measure]] = {side: deque(maxlen=HISTORY_LENGTH) for side in SIDES}
        self.last_emit = {side: -math.inf for side in SIDES}
        self.last_separation: tuple[float, np.ndarray] | None = None  # time, red-to-white offset

    def step(self, observation: Observation) -> list[Candidate]:
        """Feed one frame. Reports one frame late, since the minimum needs the frame after it."""
        self._drop_history_on_jump(observation)
        candidates = []
        for attacker_side in SIDES:
            measure = self._measure(observation, attacker_side)
            if measure is None:
                continue
            self.history[attacker_side].append(measure)
            candidate = self._check(attacker_side)
            if candidate is not None:
                candidates.append(candidate)
        return candidates

    def _drop_history_on_jump(self, observation: Observation) -> None:
        """Players cannot close faster than max_body_speed; a bigger jump means the tracker
        swapped someone in, and the tip speeds across it would be fictitious."""
        red, white = observation.players.get("red"), observation.players.get("white")
        if red is None or white is None:
            return
        separation = (red.hips - white.hips) / white.scale
        if self.last_separation is not None:
            last_time, last_offset = self.last_separation
            elapsed = observation.time - last_time
            if 0 < elapsed < 0.5:
                speed = np.linalg.norm(separation - last_offset) / elapsed
                if speed > self.config.strike.max_body_speed:
                    for history in self.history.values():
                        history.clear()
        self.last_separation = (observation.time, separation)

    def _measure(self, observation: Observation, attacker_side: str) -> _Measure | None:
        attacker = observation.players.get(attacker_side)
        defender = observation.players.get(other(attacker_side))
        if attacker is None or defender is None or attacker.shinai is None:
            return None
        zones = observation.zones[defender.side]
        gaps = strike_gaps(attacker.shinai, zones, defender.scale, self.config.shinai.monouchi_fraction)
        if not gaps:
            return None
        history = self.history[attacker_side]
        elapsed = observation.time - history[-1].time if history else 0.0
        low_frame_rate = 0 < elapsed < 0.2 and 1 / elapsed < self.config.strike.swept_below_fps
        if low_frame_rate:
            # The tip can jump clean over a zone between two frames, so also test the path it took.
            # The previous tip is stored relative to the defender; put it back into this frame's pixels.
            previous_tip = defender.hips + history[-1].tip * defender.scale
            for target in gaps:
                if target == "tsuki":
                    continue  # a thrust has to end on the target, not pass through it
                swept_gap = min(zone_gap(previous_tip, attacker.shinai.tip, zone) for zone in zones[target])
                gaps[target] = (min(gaps[target][0], swept_gap / defender.scale), gaps[target][1])
        relative_tip = (attacker.shinai.tip - defender.hips) / defender.scale
        if history and history[-1].source != attacker.shinai.source:
            # Two estimators place the tip differently; switching between them is a jump, not a motion.
            history.clear()
        return _Measure(observation.time, observation.frame_index, gaps, relative_tip, attacker.shinai.source)

    def _check(self, attacker_side: str) -> Candidate | None:
        """A candidate if the frame before the newest one was a fast arrival on a target, else None."""
        history, strike_config = self.history[attacker_side], self.config.strike
        if len(history) < 3:
            return None
        before, contact, after = history[-3], history[-2], history[-1]
        if after.time - before.time > 0.25:
            return None  # the frames are too far apart to be one motion
        if contact.time - self.last_emit[attacker_side] < strike_config.refractory_seconds:
            return None  # still the strike that was already reported
        target = min(contact.gaps, key=lambda name: contact.gaps[name][0])
        gap = contact.gaps[target][0]

        def gap_in(measure: _Measure) -> float:
            return measure.gaps.get(target, (math.inf,))[0]

        # contact: touching the zone, and no closer in the frame on either side
        if gap > strike_config.contact_distance or gap > gap_in(before) or gap > gap_in(after):
            return None
        recent = [measure for measure in history
                  if 0 <= contact.time - measure.time <= strike_config.speed_window_seconds]
        # A strike arrives from outside the zone; a shinai resting on it and jittering does not.
        widest_gap = max(measure.gaps.get(target, (gap,))[0] for measure in recent)
        if widest_gap - gap < strike_config.min_approach:
            return None
        # Speed is displacement over spans of 30 ms or more, so one frame of keypoint jitter is not read as speed.
        peak_speed = 0.0
        for measure in recent:
            elapsed = contact.time - measure.time
            if elapsed >= MIN_TIP_SPEED_SPAN:
                peak_speed = max(peak_speed, float(np.linalg.norm(contact.tip - measure.tip)) / elapsed)
        if peak_speed < strike_config.min_speed:
            return None
        # confidence in the target: how much closer it is than the next nearest one
        other_gaps = [values[0] for name, values in contact.gaps.items() if name != target]
        if other_gaps:
            runner_up = min(other_gaps)
            confidence = (runner_up + 1e-6) / (gap + runner_up + 2e-6)  # 0.5 when they tie
        else:
            confidence = 1.0
        self.last_emit[attacker_side] = contact.time
        return Candidate(contact.time, contact.frame, attacker_side, target, confidence, gap,
                         contact.gaps[target][1], peak_speed)
