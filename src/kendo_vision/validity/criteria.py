"""Yuko-datotsu criteria for one strike, each scored 0..1 or None when it cannot be observed."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import Config
from ..constants import LEFT_ANKLE, LEFT_HIP, RIGHT_ANKLE, RIGHT_HIP
from ..events.features import Observation
from ..events.heuristic import Candidate
from ..io.constants import HIGH, LOW, VOICE
from ..types import Player, other
from .constants import ARMS, MIN_BODY_SPEED_SPAN


@dataclass
class Evidence:
    candidate: Candidate
    observations: list[Observation]  # frames from just before the strike to the end of the zanshin window
    envelopes: tuple[np.ndarray, np.ndarray] | None  # audio band levels (times, levels)
    countered: bool  # the defender struck at the same time or straight after
    config: Config


def ramp(value: float, low: float, high: float) -> float:
    """0 at `low`, 1 at `high`, linear between."""
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))


def _average(scores: list[float]) -> float | None:
    return float(np.mean(scores)) if scores else None


def _series(observations: list[Observation], measure) -> tuple[np.ndarray, np.ndarray]:
    """(times, values) of measure(observation) over the frames where it is defined."""
    times, values = [], []
    for observation in observations:
        value = measure(observation)
        if value is not None:
            times.append(observation.time)
            values.append(value)
    return np.array(times), np.array(values, dtype=float)


def _track(evidence: Evidence, point_of) -> tuple[np.ndarray, np.ndarray]:
    """(times, positions) of a point on the attacker, relative to the defender's hips, in torso lengths.
    `point_of(attacker, defender)` picks the point and may return None."""
    attacker_side = evidence.candidate.attacker

    def relative_point(observation: Observation):
        attacker = observation.players.get(attacker_side)
        defender = observation.players.get(other(attacker_side))
        if attacker is None or defender is None:
            return None
        point = point_of(attacker, defender)
        return None if point is None else (point - defender.hips) / defender.scale

    return _series(evidence.observations, relative_point)


def _hips_of(attacker: Player, defender: Player):
    return attacker.hips


def _tip_of(attacker: Player, defender: Player):
    return attacker.shinai.tip if attacker.shinai is not None else None


def _lead_ankle_of(attacker: Player, defender: Player):
    """The ankle nearer the defender."""
    feet = [attacker.keypoints[index] for index in (LEFT_ANKLE, RIGHT_ANKLE) if attacker.has(index)]
    if not feet:
        return None
    return min(feet, key=lambda foot: abs(foot[0] - defender.hips[0]))


def _lean(player: Player) -> float | None:
    """Torso angle from vertical, degrees."""
    shoulders = player.shoulders
    if shoulders is None or not (player.has(LEFT_HIP) or player.has(RIGHT_HIP)):
        return None
    spine = shoulders - player.hips
    return float(np.degrees(np.arctan2(abs(spine[0]), -spine[1])))  # image y grows downward


def _angle(first: np.ndarray, vertex: np.ndarray, second: np.ndarray) -> float:
    """Angle at `vertex` between the directions to `first` and `second`, degrees."""
    to_first, to_second = first - vertex, second - vertex
    cosine = to_first @ to_second / (np.linalg.norm(to_first) * np.linalg.norm(to_second) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cosine, -1, 1))))


def _target(evidence: Evidence) -> float:
    """Datotsu-bui: how squarely the striking part met the zone."""
    return 1 - ramp(evidence.candidate.gap, 0.0, evidence.config.strike.contact_distance)


def _datotsu_bu(evidence: Evidence) -> float:
    """Contact within the monouchi (or with the tip, for tsuki)."""
    monouchi = evidence.config.shinai.monouchi_fraction
    blade_position = evidence.candidate.blade_position
    if evidence.candidate.target == "tsuki":
        return 1 - ramp(blade_position, 0.05, 0.3)
    return 1 - ramp(blade_position, monouchi, 2 * monouchi)  # falls to 0 by twice the monouchi's length


def _sae(evidence: Evidence) -> float:
    """Sharpness: fast in, abrupt stop, and an audible impact."""
    strike_time, strike_config = evidence.candidate.time, evidence.config.strike
    peak_speed = evidence.candidate.peak_speed
    scores = [ramp(peak_speed, strike_config.min_speed, 3 * strike_config.min_speed)]

    times, tips = _track(evidence, _tip_of)
    before = np.where(times <= strike_time)[0]
    just_after = np.where((times > strike_time) & (times <= strike_time + 0.12))[0]
    if len(before) and len(just_after):
        # tip speed over the 120 ms after contact, against the speed going in
        at_contact, last_after = before[-1], just_after[-1]
        speed_after = np.linalg.norm(tips[last_after] - tips[at_contact]) / (times[last_after] - times[at_contact])
        scores.append(1 - ramp(speed_after / max(peak_speed, 1e-6), 0.3, 0.9))

    if evidence.envelopes is not None:
        audio_times, high_band = evidence.envelopes[0], evidence.envelopes[1][:, HIGH]
        at_impact = high_band[np.abs(audio_times - strike_time) <= 0.06]
        baseline = high_band[(audio_times >= strike_time - 0.6) & (audio_times < strike_time - 0.1)]
        if len(at_impact) and len(baseline):
            scores.append(ramp(at_impact.max() - np.median(baseline), 6, 25))  # dB above the level just before
    return _average(scores)


def _kiai(evidence: Evidence) -> tuple[float | None, float | None]:
    """(score, onset time): voice-band level around contact against the level before it."""
    if evidence.envelopes is None:
        return None, None
    strike_time = evidence.candidate.time
    audio_times, voice_band = evidence.envelopes[0], evidence.envelopes[1][:, VOICE]
    baseline = voice_band[audio_times < strike_time - 0.6]
    around_strike = voice_band[(audio_times >= strike_time - 0.2) & (audio_times <= strike_time + 0.6)]
    if len(baseline) < 20 or len(around_strike) < 20:
        return None, None  # too little audio either side to compare
    quiet_level = np.median(baseline)
    # onset: the first moment within half a second of the strike that is clearly louder than before
    is_loud = (np.abs(audio_times - strike_time) <= 0.5) & (voice_band > quiet_level + 6)
    loud_indices = np.where(is_loud)[0]
    onset = float(audio_times[loud_indices[0]]) if len(loud_indices) else None
    return ramp(np.percentile(around_strike, 75) - quiet_level, 3, 15), onset


def _fumikomi(evidence: Evidence) -> tuple[float | None, float | None]:
    """(score, landing time): the body driving in, and when the lead foot stops."""
    strike_time = evidence.candidate.time
    times, hip_offsets = _track(evidence, _hips_of)
    in_lunge = (times >= strike_time - 0.35) & (times <= strike_time + 0.05)
    if in_lunge.sum() < 3:
        return None, None
    times, separation = times[in_lunge], np.linalg.norm(hip_offsets[in_lunge], axis=1)
    # Fastest change in the players' separation over any pair of frames far enough apart.
    # Either direction counts, because hiki-waza is struck while moving away.
    body_speed = 0.0
    for earlier in range(len(times)):
        for later in range(earlier + 1, len(times)):
            span = times[later] - times[earlier]
            if span >= MIN_BODY_SPEED_SPAN:
                body_speed = max(body_speed, abs(separation[later] - separation[earlier]) / span)
    return ramp(body_speed, 1.0, 4.0), _foot_landing(evidence)


def _foot_landing(evidence: Evidence) -> float | None:
    """When the lead foot comes down: from the ankle's motion, else from the stomp in the audio."""
    strike_time = evidence.candidate.time
    times, positions = _track(evidence, _lead_ankle_of)
    around_strike = (times >= strike_time - 0.35) & (times <= strike_time + 0.25)
    if around_strike.sum() >= 4:
        times, positions = times[around_strike], positions[around_strike]
        speeds = np.linalg.norm(np.diff(positions, axis=0), axis=1) / np.diff(times)
        fastest = int(speeds.argmax())
        # the foot has landed at the first frame, after its fastest, where it has all but stopped
        stopped = np.where(speeds[fastest:] < 0.25 * speeds[fastest])[0]
        if speeds[fastest] >= 2.0 and len(stopped):
            return float(times[fastest + stopped[0]])
    if evidence.envelopes is not None:
        audio_times, low_band = evidence.envelopes[0], evidence.envelopes[1][:, LOW]
        near_strike = np.abs(audio_times - strike_time) <= 0.2
        baseline = low_band[(audio_times >= strike_time - 0.8) & (audio_times < strike_time - 0.25)]
        if near_strike.any() and len(baseline) and low_band[near_strike].max() > np.median(baseline) + 8:
            return float(audio_times[near_strike][low_band[near_strike].argmax()])  # the loudest thump
    return None


def _sync(evidence: Evidence, foot_time: float | None, kiai_score: float | None,
          kiai_onset: float | None) -> float | None:
    """Ki-ken-tai-ichi: foot landing and kiai coinciding with contact."""
    strike_time = evidence.candidate.time
    frame_times = [observation.time for observation in evidence.observations]
    frame_interval = float(np.median(np.diff(frame_times))) if len(frame_times) > 1 else 1 / 30
    tolerance = 0.06 + frame_interval  # timing cannot be resolved finer than a frame
    scores = []
    if foot_time is not None:
        scores.append(1 - ramp(abs(foot_time - strike_time), tolerance, 0.25))
    if kiai_score is not None:  # there was audio to listen to
        if kiai_onset is None:
            scores.append(0.0)  # no shout anywhere near the strike
        else:
            lead = strike_time - kiai_onset  # positive: the kiai began before contact
            if lead >= 0:
                scores.append(1 - ramp(lead, 0.4, 0.8))  # starting up to 0.4 s early is fine
            else:
                scores.append(1 - ramp(-lead, tolerance, 0.25))  # a late kiai is judged like a late foot
    return _average(scores)


def _posture(evidence: Evidence) -> float | None:
    """Shisei at contact: upright torso and extended arms."""
    attacker_side, strike_time = evidence.candidate.attacker, evidence.candidate.time
    leans, elbow_angles = [], []
    for observation in evidence.observations:
        attacker = observation.players.get(attacker_side)
        if attacker is None or abs(observation.time - strike_time) > 0.05:
            continue  # only the frames right at contact
        lean = _lean(attacker)
        if lean is not None:
            leans.append(lean)
        for shoulder, elbow, wrist in ARMS:
            if attacker.has(shoulder, elbow, wrist):
                elbow_angles.append(_angle(attacker.keypoints[shoulder], attacker.keypoints[elbow],
                                           attacker.keypoints[wrist]))
    upright_score = 1 - ramp(np.mean(leans), 20, 45) if leans else None
    arms_score = ramp(np.mean(elbow_angles), 100, 150) if elbow_angles else None
    if upright_score is None or arms_score is None:
        return upright_score if arms_score is None else arms_score  # whichever one was measurable
    return 0.6 * upright_score + 0.4 * arms_score


def _zanshin(evidence: Evidence) -> float | None:
    """After the strike: stays up, carries through, and ends with the shinai back on the opponent."""
    strike_time, attacker_side = evidence.candidate.time, evidence.candidate.attacker
    window_end = strike_time + evidence.config.validity.zanshin_seconds
    after_strike = [observation for observation in evidence.observations
                    if strike_time + 0.15 <= observation.time <= window_end]
    if len(after_strike) < 3 or after_strike[-1].time < window_end - 0.5:
        return None  # the stream ended before the window did

    # 1. Stays on their feet and in view.
    upright_frames = 0
    for observation in after_strike:
        attacker = observation.players.get(attacker_side)
        lean = _lean(attacker) if attacker is not None else None
        if lean is not None and lean < 60:
            upright_frames += 1
    scores = [ramp(upright_frames / len(after_strike), 0.5, 0.9)]

    # 2. Carries through: how far the body travels in the 0.8 s after contact.
    times, hip_offsets = _track(evidence, _hips_of)
    up_to_contact = np.where(times <= strike_time)[0]
    carrying_through = (times > strike_time) & (times <= strike_time + 0.8)
    if len(up_to_contact) and carrying_through.any():
        position_at_contact = hip_offsets[up_to_contact[-1]]
        travelled = np.linalg.norm(hip_offsets[carrying_through] - position_at_contact, axis=1).max()
        scores.append(ramp(travelled, 0.3, 1.2))

    # 3. Ends ready: over the last 0.5 s the shinai points back at the opponent.
    aim_errors = []
    for observation in after_strike:
        attacker = observation.players.get(attacker_side)
        defender = observation.players.get(other(attacker_side))
        if observation.time < window_end - 0.5 or attacker is None or defender is None or attacker.shinai is None:
            continue
        toward_opponent = attacker.shinai.butt + defender.hips - attacker.hips
        aim_errors.append(_angle(attacker.shinai.tip, attacker.shinai.butt, toward_opponent))
    if aim_errors:
        scores.append(1 - ramp(np.mean(aim_errors), 45, 110))  # degrees off the line to the opponent

    return _average(scores) * (0.5 if evidence.countered else 1.0)


def score(evidence: Evidence) -> dict[str, float | None]:
    kiai_score, kiai_onset = _kiai(evidence)
    fumikomi_score, foot_time = _fumikomi(evidence)
    return {
        "target": _target(evidence), "datotsu_bu": _datotsu_bu(evidence), "sae": _sae(evidence),
        "kiai": kiai_score, "fumikomi": fumikomi_score,
        "sync": _sync(evidence, foot_time, kiai_score, kiai_onset),
        "posture": _posture(evidence), "zanshin": _zanshin(evidence),
    }
