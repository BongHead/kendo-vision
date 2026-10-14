"""Draws players, shinai, target zones, the score and the latest strike onto a frame."""
from __future__ import annotations

import cv2
import numpy as np

from ..constants import KEYPOINT_COUNT, SIDES
from ..events.features import Observation
from ..events.heuristic import Candidate
from ..match import MatchState
from ..types import StrikeEvent
from .constants import BANNER_HOLD_SECONDS, COLORS, FONT, REFERENCE_HEIGHT, SHINAI_COLOR, SKELETON, ZONE_COLOR


def _pixel(point) -> tuple[int, int]:
    return int(round(point[0])), int(round(point[1]))


class Overlay:
    def __init__(self, hold_seconds: float = BANNER_HOLD_SECONDS):
        self.hold_seconds = hold_seconds  # how long a banner stays up
        self.banner_lines: list[str] = []
        self.banner_color = (255, 255, 255)
        self.banner_time = float("-inf")

    def strike(self, candidate: Candidate) -> None:
        """Announce a strike that is still being judged."""
        self.banner_lines = [f"{candidate.target.upper()}  {candidate.attacker}  judging..."]
        self.banner_color, self.banner_time = COLORS[candidate.attacker], candidate.time

    def verdict(self, event: StrikeEvent, now: float) -> None:
        headline = f"{event.target.upper()}  {event.attacker}  {event.verdict.upper()}  {event.validity:.2f}"
        scores = [f"{name} {'--' if value is None else f'{value:.2f}'}" for name, value in event.criteria.items()]
        # the eight criteria go on two lines of four
        self.banner_lines = [headline, "  ".join(scores[:4]), "  ".join(scores[4:])]
        self.banner_color, self.banner_time = COLORS[event.attacker], now

    def draw(self, image: np.ndarray, observation: Observation, match: MatchState, fps: float) -> np.ndarray:
        height = image.shape[0]
        size = height / REFERENCE_HEIGHT  # text and lines grow with resolution
        thickness = max(1, round(2 * size))
        for side, player in observation.players.items():
            color = COLORS[side]
            for start, end in SKELETON:
                if player.has(start, end):
                    cv2.line(image, _pixel(player.keypoints[start]), _pixel(player.keypoints[end]),
                             color, thickness, cv2.LINE_AA)
            for index in range(KEYPOINT_COUNT):
                if player.has(index):
                    cv2.circle(image, _pixel(player.keypoints[index]), thickness + 1, color, -1, cv2.LINE_AA)
            for zones in observation.zones[side].values():
                for zone in zones:
                    if np.array_equal(zone.start, zone.end):  # a circular zone (men, tsuki)
                        cv2.circle(image, _pixel(zone.start), int(zone.radius), ZONE_COLOR, 1, cv2.LINE_AA)
                    else:  # a capsule (kote, do), drawn as a thick line
                        cv2.line(image, _pixel(zone.start), _pixel(zone.end), ZONE_COLOR,
                                 max(1, int(zone.radius / 2)), cv2.LINE_AA)
            if player.shinai is not None:
                cv2.line(image, _pixel(player.shinai.butt), _pixel(player.shinai.tip), SHINAI_COLOR,
                         thickness, cv2.LINE_AA)
                cv2.circle(image, _pixel(player.shinai.tip), thickness + 2, color, -1, cv2.LINE_AA)
            if player.box is not None:
                label_position = (player.box[0], player.box[1] - 8 * size)  # just above the box
                cv2.putText(image, side, _pixel(label_position), FONT, 0.6 * size, color, thickness, cv2.LINE_AA)

        scoreboard = "   ".join(f"{side.upper()} {match.score[side]}" for side in SIDES)
        if match.winner:
            scoreboard += f"   winner: {match.winner}"
        self._text(image, f"{scoreboard}   {fps:.0f} fps", (10 * size, 28 * size), 0.8 * size,
                   (255, 255, 255), thickness)

        banner_age = observation.time - self.banner_time
        if self.banner_lines and 0 <= banner_age <= self.hold_seconds:
            for line_index, line in enumerate(self.banner_lines):
                is_headline = line_index == 0
                position = (10 * size, height - (70 - 26 * line_index) * size)  # stacked up from the bottom
                self._text(image, line, position, (0.9 if is_headline else 0.5) * size, self.banner_color,
                           thickness if is_headline else max(1, thickness - 1))
        return image

    @staticmethod
    def _text(image, text, position, scale, color, thickness) -> None:
        """Text with a dark outline so it reads on any background."""
        cv2.putText(image, text, _pixel(position), FONT, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(image, text, _pixel(position), FONT, scale, color, thickness, cv2.LINE_AA)
