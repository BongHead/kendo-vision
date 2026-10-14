"""Turns strike candidates into judged events once their zanshin window has passed."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import Config, ValidityConfig
from ..events.features import FeatureBuffer
from ..events.heuristic import Candidate
from ..types import StrikeEvent
from .constants import LOWEST_SCORE, RECENT_SECONDS
from .criteria import Evidence, score


def rubric(criteria: dict[str, float | None], validity_config: ValidityConfig) -> float:
    """Weighted geometric mean of the observable criteria; target and datotsu-bu are gates."""
    if min(criteria["target"], criteria["datotsu_bu"]) < validity_config.gate:
        return 0.0
    weighted_log_sum = total_weight = 0.0
    for name, criterion_score in criteria.items():
        weight = validity_config.weights.get(name, 0.0)
        if criterion_score is None or weight <= 0:
            continue  # unobservable criteria neither help nor hurt
        weighted_log_sum += weight * np.log(max(criterion_score, LOWEST_SCORE))
        total_weight += weight
    return float(np.exp(weighted_log_sum / total_weight)) if total_weight else 0.0


class ValidityJudge:
    def __init__(self, config: Config, audio=None):
        self.config, self.audio = config, audio
        self.pending: list[Candidate] = []
        self.recent: list[Candidate] = []  # for spotting counters and ai-uchi
        self.classifier = None
        if config.validity.mode == "learned":
            weights = config.model.strikenet_weights
            if not Path(weights).exists():
                raise FileNotFoundError(f"validity.mode is 'learned' but {weights} is missing")
            from ..events.strikenet import StrikeClassifier

            self.classifier = StrikeClassifier(config)

    def push(self, candidate: Candidate) -> None:
        self.pending.append(candidate)
        still_recent = [earlier for earlier in self.recent if candidate.time - earlier.time < RECENT_SECONDS]
        self.recent = still_recent + [candidate]

    def poll(self, buffer: FeatureBuffer, now: float, final: bool = False) -> list[StrikeEvent]:
        """Events whose wait is over. `final` judges whatever is left at the end of the stream."""
        wait = self.config.validity.zanshin_seconds
        ready, waiting = [], []
        for candidate in self.pending:
            if final or now >= candidate.time + wait:
                ready.append(candidate)
            else:
                waiting.append(candidate)
        self.pending = waiting
        events = []
        for candidate in ready:
            event = self._judge(candidate, buffer)
            if event is not None:
                events.append(event)
        return events

    def _judge(self, candidate: Candidate, buffer: FeatureBuffer) -> StrikeEvent | None:
        validity_config = self.config.validity
        # a strike by the other player at the same moment (ai-uchi) or just after (a counter)
        countered = any(
            rival.attacker != candidate.attacker
            and -0.05 <= rival.time - candidate.time <= validity_config.counter_seconds
            for rival in self.recent
        )
        envelopes = None
        if self.audio is not None:
            envelopes = self.audio.envelopes(candidate.time - 2.6, candidate.time + 0.7)
        observations = buffer.span(candidate.time - 0.6, candidate.time + validity_config.zanshin_seconds)
        criteria = score(Evidence(candidate, observations, envelopes, countered, self.config))
        event = StrikeEvent(candidate.time, candidate.frame, candidate.attacker, candidate.target,
                            candidate.target_confidence, criteria)
        if self.classifier is not None:
            target, confidence, attacker, ippon_probability = self.classifier(buffer, candidate.time, self.audio)
            if target is None:
                return None  # the network says this was not a strike
            event.target, event.target_confidence, event.attacker = target, confidence, attacker
            event.validity = ippon_probability
        else:
            event.validity = rubric(criteria, validity_config)
        if event.validity >= validity_config.ippon:
            event.verdict = "ippon"
        elif event.validity >= validity_config.borderline:
            event.verdict = "borderline"
        else:
            event.verdict = "no"
        return event
