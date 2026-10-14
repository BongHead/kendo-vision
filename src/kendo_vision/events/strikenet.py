"""Small temporal network over feature windows, with target, attacker and validity heads."""
from __future__ import annotations

import torch
from torch import nn

from ..config import Config
from ..constants import SIDES
from .constants import CLASSES, DILATIONS, FEATURE_SIZE
from .features import FeatureBuffer, build_window, static_features


class _Block(nn.Module):
    """One dilated convolution with a skip connection."""

    def __init__(self, width: int, dilation: int, dropout: float):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation),
            nn.BatchNorm1d(width), nn.ReLU(), nn.Dropout(dropout),
        )

    def forward(self, hidden):
        return hidden + self.layers(hidden)


class StrikeNet(nn.Module):
    """Dilated 1D convolutions over time, pooled, then three linear heads."""

    def __init__(self, input_size: int = FEATURE_SIZE, width: int = 128, dropout: float = 0.2):
        super().__init__()
        self.input_layer = nn.Conv1d(input_size, width, 1)
        self.blocks = nn.Sequential(*[_Block(width, dilation, dropout) for dilation in DILATIONS])
        self.target_head = nn.Linear(2 * width, len(CLASSES))
        self.attacker_head = nn.Linear(2 * width, len(SIDES))
        self.validity_head = nn.Linear(2 * width, 1)

    def forward(self, windows):
        """(batch, steps, features) -> target logits, attacker logits, ippon logit."""
        hidden = self.blocks(self.input_layer(windows.transpose(1, 2)))  # Conv1d wants (batch, features, steps)
        # summarise the whole window: its average and its strongest response per channel
        pooled = torch.cat([hidden.mean(dim=-1), hidden.amax(dim=-1)], dim=1)
        return self.target_head(pooled), self.attacker_head(pooled), self.validity_head(pooled).squeeze(-1)


class StrikeClassifier:
    """Runs StrikeNet on the window around a candidate strike."""

    def __init__(self, config: Config):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.network = StrikeNet().to(self.device).eval()
        self.network.load_state_dict(torch.load(config.model.strikenet_weights, map_location=self.device))

    @torch.no_grad()
    def __call__(self, buffer: FeatureBuffer, strike_time: float, audio) -> tuple[str | None, float, str, float]:
        """(target or None if not a strike, target confidence, attacker, P(ippon))."""
        net = self.config.net
        times, raw_rows = buffer.raw()
        envelopes = None
        if audio is not None:
            envelopes = audio.envelopes(strike_time - net.before_seconds, strike_time + net.after_seconds)
        window = build_window(times, static_features(raw_rows, self.config), strike_time, self.config, envelopes)
        batch = torch.from_numpy(window)[None].to(self.device)  # a batch of one
        target_logits, attacker_logits, ippon_logit = self.network(batch)
        probabilities = target_logits.softmax(dim=-1)[0]
        best_class = int(probabilities.argmax())
        target = CLASSES[best_class] if best_class else None  # class 0 is "none"
        attacker = SIDES[int(attacker_logits.argmax())]
        return target, float(probabilities[best_class]), attacker, float(ippon_logit.sigmoid())
