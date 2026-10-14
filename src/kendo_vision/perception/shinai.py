"""Shinai keypoints (butt, tsuba, tip) per player: a wrist-based baseline and a trained model."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from ..config import Config, ModelConfig, ShinaiConfig
from ..constants import LEFT_WRIST, RIGHT_WRIST
from ..types import Player, Shinai
from .constants import (EXPECTED_BLADE_LENGTH, MAX_BLADE_LENGTH, MAX_GRIP_DISTANCE, MAX_SHINAI_MATCH_COST,
                        MIN_BLADE_ELONGATION, SHINAI_PROMPTS, TSUBA_POSITION, TSUKA_TO_BLADE)
from .pose import predict_args


class WristShinaiEstimator:
    """Extends the wrist line: the left hand holds the butt, the right hand sits below the tsuba."""

    def __init__(self, config: ShinaiConfig):
        self.config = config
        self.last_direction: dict[int, np.ndarray] = {}  # by player slot

    def reset(self) -> None:
        """Forget direction history; call between unrelated images."""
        self.last_direction.clear()

    def __call__(self, image: np.ndarray, players: list[Player]) -> None:
        for player in players:
            player.shinai = self._estimate(player)

    def _estimate(self, player: Player) -> Shinai | None:
        if not player.has(LEFT_WRIST, RIGHT_WRIST):
            return None
        config = self.config
        left_wrist, right_wrist = player.keypoints[LEFT_WRIST], player.keypoints[RIGHT_WRIST]
        separation = float(np.linalg.norm(right_wrist - left_wrist))
        previous = self.last_direction.get(player.slot)
        confidence = float(min(player.keypoint_confidence[LEFT_WRIST], player.keypoint_confidence[RIGHT_WRIST]))
        butt = left_wrist
        if separation < config.min_hand_separation * player.scale:
            # hands overlap in the image, so their line is noise: keep the last direction
            if previous is None:
                return None
            direction, confidence = previous, 0.3 * confidence
        else:
            direction = (right_wrist - left_wrist) / separation
            if previous is not None and direction @ previous < -0.85:
                # a 180 degree flip in one frame is the pose model swapping left and right wrists
                direction, butt = -direction, right_wrist
            self.last_direction[player.slot] = direction
        # Hands and tip are collinear, so foreshortening scales them alike; cap by body size.
        length = min(config.hand_ratio * separation, config.length_ratio * player.scale)
        return Shinai(butt=butt, tsuba=butt + TSUBA_POSITION * length * direction,
                      tip=butt + length * direction, confidence=confidence, source="wrist")


class ShinaiKeypointModel:
    """Trained 3-keypoint detector; detections go to the player whose wrists they sit on."""

    def __init__(self, model_config: ModelConfig, fallback: WristShinaiEstimator):
        from ultralytics import YOLO

        self.model = YOLO(model_config.shinai_weights)
        self.arguments = predict_args(model_config, model_config.shinai_image_size, model_config.shinai_confidence)
        self.fallback = fallback

    def reset(self) -> None:
        self.fallback.reset()

    def __call__(self, image: np.ndarray, players: list[Player]) -> None:
        result = self.model.predict(image, **self.arguments)[0]
        detected = []
        if result.keypoints is not None and result.boxes is not None and len(result.boxes):
            keypoints = result.keypoints.xy.cpu().numpy().astype(float)
            scores = result.boxes.conf.cpu().numpy()
            detected = [Shinai(points[0], points[1], points[2], float(score))
                        for points, score in zip(keypoints, scores)]
        for player in players:
            player.shinai = None
        # Hand out detections greedily, cheapest (closest to a player's hands) first.
        pairings = sorted((self._cost(shinai, player), shinai_index, player_index)
                          for shinai_index, shinai in enumerate(detected)
                          for player_index, player in enumerate(players))
        taken: set[int] = set()
        for cost, shinai_index, player_index in pairings:
            player = players[player_index]
            if cost < MAX_SHINAI_MATCH_COST and shinai_index not in taken and player.shinai is None:
                player.shinai = detected[shinai_index]
                taken.add(shinai_index)
        # anyone the model missed gets the wrist estimate instead
        self.fallback(image, [player for player in players if player.shinai is None])

    @staticmethod
    def _cost(shinai: Shinai, player: Player) -> float:
        """How far the shinai's grip is from the player's hands, in torso lengths."""
        if player.has(LEFT_WRIST, RIGHT_WRIST):
            butt_to_left_hand = np.linalg.norm(shinai.butt - player.keypoints[LEFT_WRIST])
            tsuba_to_right_hand = np.linalg.norm(shinai.tsuba - player.keypoints[RIGHT_WRIST])
            return float(butt_to_left_hand + tsuba_to_right_hand) / (2 * player.scale)
        # wrists not visible: fall back to how near the grip is to the body
        grip = (shinai.butt + shinai.tsuba) / 2
        return float(np.linalg.norm(grip - player.hips)) / (2 * player.scale)


def _load_prompted_yoloe(model_config: ModelConfig):
    """YOLOE with the shinai prompts built in, created and saved on first use."""
    from ultralytics import YOLOE

    prompted = Path(model_config.yoloe_prompted_weights)
    if prompted.exists():
        return YOLOE(str(prompted))
    model = YOLOE(model_config.yoloe_weights)
    prompts = list(SHINAI_PROMPTS)
    # Turning the prompts into class embeddings needs a text encoder, which Ultralytics downloads into
    # the working directory; do that step inside the models folder so the download lands there.
    previous_folder = os.getcwd()
    prompted.parent.mkdir(parents=True, exist_ok=True)
    os.chdir(prompted.parent)
    try:
        model.set_classes(prompts, model.get_text_pe(prompts))
    finally:
        os.chdir(previous_folder)
    model.save(str(prompted))
    return model


def _blade_segment(outline: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """The two ends of a mask outline's long axis, or None if the mask is not long and thin."""
    if len(outline) < 3:
        return None
    center = outline.mean(axis=0)
    # principal axes of the outline points: the first is along the blade, the second across it
    _, _, axes = np.linalg.svd(outline - center, full_matrices=False)
    along = (outline - center) @ axes[0]
    across = np.abs((outline - center) @ axes[1])
    length, width = along.max() - along.min(), 2 * across.mean() + 1e-6
    if length / width < MIN_BLADE_ELONGATION:
        return None
    return center + along.min() * axes[0], center + along.max() * axes[0]


def _hands(player: Player) -> np.ndarray:
    """Where the player's grip is: between the wrists, or the torso centre when they are not visible."""
    wrists = [player.keypoints[index] for index in (LEFT_WRIST, RIGHT_WRIST) if player.has(index)]
    if wrists:
        return np.mean(wrists, axis=0)
    shoulders = player.shoulders
    return player.hips if shoulders is None else (shoulders + player.hips) / 2


class OpenVocabularyShinai:
    """Finds shinai from a text prompt with YOLOE, so no training is needed; each mask gives a blade line."""

    def __init__(self, model_config: ModelConfig, fallback: WristShinaiEstimator):
        self.model = _load_prompted_yoloe(model_config)
        self.arguments = predict_args(model_config, model_config.shinai_image_size, model_config.yoloe_confidence)
        self.fallback = fallback

    def reset(self) -> None:
        self.fallback.reset()

    def __call__(self, image: np.ndarray, players: list[Player]) -> None:
        result = self.model.predict(image, **self.arguments)[0]
        blades = []  # (end, other end, confidence)
        if result.masks is not None:
            for outline, score in zip(result.masks.xy, result.boxes.conf.cpu().numpy()):
                ends = _blade_segment(np.asarray(outline, float))
                if ends is not None:
                    blades.append((ends[0], ends[1], float(score)))
        for player in players:
            player.shinai = None
        # Each blade goes to the player whose hands are nearest one of its ends; that end is the tsuba side.
        pairings = []  # (distance in torso lengths, blade index, player index, near end, far end)
        for blade_index, (first_end, second_end, _) in enumerate(blades):
            for player_index, player in enumerate(players):
                hands = _hands(player)
                for near_end, far_end in ((first_end, second_end), (second_end, first_end)):
                    distance = float(np.linalg.norm(near_end - hands)) / player.scale
                    pairings.append((distance, blade_index, player_index, near_end, far_end))
        pairings.sort(key=lambda pairing: pairing[0])
        used_ends: set[tuple[int, int]] = set()  # (blade index, which end) already given to a player
        for distance, blade_index, player_index, near_end, far_end in pairings:
            player = players[player_index]
            if distance > MAX_GRIP_DISTANCE or player.shinai is not None:
                continue
            end_id = (blade_index, int(near_end is blades[blade_index][1]))
            if end_id in used_ends:
                continue
            blade = far_end - near_end
            longest_blade = MAX_BLADE_LENGTH * player.scale
            blade_length = float(np.linalg.norm(blade))
            if blade_length > longest_blade:
                # Longer than any blade: two crossed shinai merged into one mask. Each player keeps
                # a blade's length from their own end, and the other end stays free for the opponent.
                far_end = near_end + blade * (EXPECTED_BLADE_LENGTH * player.scale / blade_length)
            else:
                used_ends.update({(blade_index, 0), (blade_index, 1)})  # a single blade belongs to one player
            # the mask covers the blade; the handle, hidden by the gloves, extends behind the tsuba
            butt = near_end - TSUKA_TO_BLADE * (far_end - near_end)
            used_ends.add(end_id)
            player.shinai = Shinai(butt=butt, tsuba=near_end, tip=far_end, confidence=blades[blade_index][2],
                                   source="yoloe")
        self.fallback(image, [player for player in players if player.shinai is None])


def make_shinai_estimator(config: Config):
    """The trained model when its weights exist, else YOLOE or the wrist baseline as configured."""
    baseline = WristShinaiEstimator(config.shinai)
    if Path(config.model.shinai_weights).exists():
        return ShinaiKeypointModel(config.model, baseline)
    if config.model.shinai_finder == "yoloe":
        return OpenVocabularyShinai(config.model, baseline)
    return baseline
