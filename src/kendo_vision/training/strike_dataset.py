"""PyTorch dataset of feature windows around labelled strikes and background moments."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..config import Config
from ..constants import SIDES
from ..events.constants import CLASSES, SWAP_SIDES, X_COLUMNS, XY_COLUMNS
from ..events.features import build_window, static_features
from ..io.sink import read_jsonl
from .constants import COORDINATE_NOISE, NONE_CLASS, TIME_JITTER, UNKNOWN


class StrikeDataset(Dataset):
    """A recording is <name>.npz from `run --dump-features` plus <name>.jsonl from `label-events`.

    Yields (window, target class, attacker, ippon); attacker and ippon are UNKNOWN where unlabelled.
    """

    def __init__(self, names: list[str], folder: str | Path, config: Config, train: bool):
        self.config, self.train = config, train
        self.recordings = []  # per recording: (times, static features, audio envelopes, event times)
        self.items = []  # per sample: (recording index, time or None for "pick a background moment", labels...)
        for recording_index, name in enumerate(names):
            archive = np.load(Path(folder) / f"{name}.npz")
            envelopes = (archive["audio_times"], archive["audio_levels"]) if "audio_times" in archive else None
            events = read_jsonl(Path(folder) / f"{name}.jsonl")
            event_times = np.array([event["time"] for event in events])
            self.recordings.append((archive["times"], static_features(archive["raw"], config), envelopes, event_times))
            strike_count = 0
            for event in events:
                target_class = CLASSES.index(event.get("target") or "none")
                if target_class == NONE_CLASS:
                    # a candidate the labeller rejected: a hard example of "not a strike"
                    self.items.append((recording_index, event["time"], NONE_CLASS, UNKNOWN, float(UNKNOWN)))
                    continue
                ippon = float(UNKNOWN) if event.get("ippon") is None else float(event["ippon"])
                self.items.append((recording_index, event["time"], target_class, SIDES.index(event["attacker"]), ippon))
                strike_count += 1
            # balance the strikes with as many windows from moments where nothing happens
            self.items += [(recording_index, None, NONE_CLASS, UNKNOWN, float(UNKNOWN))] * strike_count
        self.random = np.random.default_rng(0)

    def __len__(self) -> int:
        return len(self.items)

    def _background_time(self, times: np.ndarray, event_times: np.ndarray, random) -> float:
        """A moment at least a second away from every labelled event."""
        net = self.config.net
        earliest = times[0] + net.before_seconds
        latest = max(times[-1] - net.after_seconds, earliest)
        for _ in range(20):  # a few tries; a recording packed with events keeps the last draw
            moment = random.uniform(earliest, latest)
            if not len(event_times) or np.abs(event_times - moment).min() > 1.0:
                break
        return float(moment)

    def __getitem__(self, index: int):
        recording_index, center_time, target_class, attacker, ippon = self.items[index]
        times, static, envelopes, event_times = self.recordings[recording_index]
        # validation uses a generator seeded by the index, so its windows are the same every epoch
        random = self.random if self.train else np.random.default_rng(index)
        if center_time is None:
            center_time = self._background_time(times, event_times, random)
        elif self.train:
            center_time += random.uniform(-TIME_JITTER, TIME_JITTER)
        swap_sides = self.train and random.random() < 0.5  # exchange red and white
        mirror = self.train and random.random() < 0.5  # the same match seen from the other side of the court

        def augment(window: np.ndarray) -> np.ndarray:
            if swap_sides:
                window = window[:, SWAP_SIDES]
            if self.train:
                window = window.copy()
                if mirror:
                    window[:, X_COLUMNS] *= -1
                noise = random.normal(0, COORDINATE_NOISE, (len(window), len(XY_COLUMNS)))
                window[:, XY_COLUMNS] += noise.astype(np.float32)
            return window

        window = build_window(times, static, center_time, self.config, envelopes, augment)
        if swap_sides and attacker != UNKNOWN:
            attacker = 1 - attacker  # the attacker label has to follow the swap
        return torch.from_numpy(window), target_class, attacker, np.float32(ippon)
