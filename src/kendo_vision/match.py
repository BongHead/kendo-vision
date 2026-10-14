"""Match score."""
from __future__ import annotations

from .config import MatchConfig
from .constants import SIDES
from .types import StrikeEvent


class MatchState:
    def __init__(self, config: MatchConfig):
        self.config = config
        self.score = {side: 0 for side in SIDES}
        self.winner: str | None = None
        self.lockout_until = float("-inf")

    def apply(self, event: StrikeEvent) -> None:
        """Count an ippon unless the match is over or play had stopped for the previous point."""
        if event.verdict != "ippon" or self.winner or event.time < self.lockout_until:
            return
        self.score[event.attacker] += 1
        event.counted = True
        self.lockout_until = event.time + self.config.lockout_seconds
        if self.score[event.attacker] >= self.config.points_to_win:
            self.winner = event.attacker
