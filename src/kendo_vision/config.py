"""Configuration dataclasses; a YAML file overrides the defaults here."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path

import yaml

# Distances are in torso lengths (shoulder midpoint to hip midpoint) unless noted.


@dataclass
class ModelConfig:
    pose_weights: str = "models/yolo26s-pose.pt"  # downloaded on first use
    shinai_weights: str = "models/shinai.pt"  # trained shinai model; used whenever the file exists
    shinai_finder: str = "yoloe"  # without a trained model: yoloe (text-prompted detector) or wrist
    yoloe_weights: str = "models/yoloe-11l-seg.pt"  # downloaded on first use
    yoloe_prompted_weights: str = "models/shinai-yoloe.pt"  # yoloe with the shinai prompts built in, made once
    yoloe_confidence: float = 0.05  # yoloe scores shinai low even when right
    strikenet_weights: str = "models/strikenet.pt"
    image_size: int = 960  # longest side the pose model sees, pixels
    shinai_image_size: int = 1280
    pose_confidence: float = 0.35  # detections below this are dropped
    shinai_confidence: float = 0.25
    device: str = ""  # "" picks the GPU when available
    half_precision: bool = True  # FP16 on GPU


@dataclass
class CameraConfig:
    """Requested live-capture format; the camera may give something else."""
    width: int = 1280
    height: int = 720
    fps: int = 60


@dataclass
class TrackConfig:
    tracker: str = "botsort.yaml"  # bytetrack.yaml is faster, with weaker ids through crossings
    red_side: str = "left"  # where red stands when tracking starts
    use_tasuki: bool = False  # let tasuki colour override the start position
    tasuki_margin: float = 0.02  # red-pixel fraction gap needed to override
    max_lost_seconds: float = 2.0  # both players lost this long -> start over
    gate: float = 0.8  # max match distance, in box heights
    filter_players: bool = True  # drop shinpan and spectators with a zero-shot image classifier
    player_threshold: float = 0.5  # how sure the classifier must be that someone is a kendoka
    reclassify_every: int = 15  # frames between re-checks of the same tracked person


@dataclass
class ShinaiConfig:
    length_ratio: float = 2.3  # shinai length / torso length
    hand_ratio: float = 4.5  # (left hand to tip) / hand separation
    min_hand_separation: float = 0.12  # below this the wrist line is unreliable
    monouchi_fraction: float = 0.33  # share of the blade, from the tip, that counts as monouchi


@dataclass
class ZoneConfig:
    men_radius: float = 0.30
    tsuki_radius: float = 0.09
    kote_radius: float = 0.09
    kote_length: float = 0.5  # share of the forearm, from the wrist
    do_radius: float = 0.11
    do_low: float = 0.10  # do spans this share of hip->shoulder...
    do_high: float = 0.60  # ...up to this share


@dataclass
class StrikeConfig:
    contact_distance: float = 0.12  # monouchi within this of a zone counts as contact
    min_speed: float = 8.0  # tip speed, torso lengths per second
    min_approach: float = 0.4  # the gap must have closed by this much within speed_window_seconds
    max_body_speed: float = 20.0  # players closing faster than this is a tracking jump, not motion
    speed_window_seconds: float = 0.15
    refractory_seconds: float = 0.5  # one strike cannot be reported twice within this
    swept_below_fps: float = 45.0  # also test the tip's path between frames at low frame rates
    buffer_seconds: float = 6.0  # how much history is kept for judging


@dataclass
class AudioConfig:
    enabled: bool = True
    sample_rate: int = 16000
    offset_seconds: float = 0.0  # added to audio time to line it up with video
    mic_device: int | None = None


@dataclass
class ValidityConfig:
    mode: str = "rubric"  # rubric | learned (needs strikenet weights)
    zanshin_seconds: float = 1.8  # how long after contact to watch before judging
    gate: float = 0.3  # target and datotsu_bu must reach this
    ippon: float = 0.65
    borderline: float = 0.45
    counter_seconds: float = 0.4  # a defender strike this soon after halves zanshin
    weights: dict = field(default_factory=lambda: {
        "target": 1.0, "datotsu_bu": 1.0, "sae": 1.0, "kiai": 0.7,
        "fumikomi": 0.8, "sync": 0.8, "posture": 0.7, "zanshin": 1.0,
    })


@dataclass
class MatchConfig:
    points_to_win: int = 2
    lockout_seconds: float = 3.0  # ignore strikes this long after a counted ippon


@dataclass
class NetConfig:
    """The window StrikeNet sees around each strike."""
    rate_hz: int = 60  # features are resampled to this rate
    before_seconds: float = 0.5
    after_seconds: float = 2.0


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    track: TrackConfig = field(default_factory=TrackConfig)
    shinai: ShinaiConfig = field(default_factory=ShinaiConfig)
    zones: ZoneConfig = field(default_factory=ZoneConfig)
    strike: StrikeConfig = field(default_factory=StrikeConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    validity: ValidityConfig = field(default_factory=ValidityConfig)
    match: MatchConfig = field(default_factory=MatchConfig)
    net: NetConfig = field(default_factory=NetConfig)


def _merge(section, overrides: dict, prefix: str = "") -> None:
    """Copy `overrides` onto a config dataclass, descending into nested sections."""
    for key, value in overrides.items():
        if not hasattr(section, key):
            raise ValueError(f"Unknown config key: {prefix}{key}")
        current = getattr(section, key)
        if is_dataclass(current):
            _merge(current, value or {}, f"{prefix}{key}.")
        elif isinstance(current, dict):
            current.update(value or {})  # e.g. validity.weights: override only the named entries
        else:
            setattr(section, key, value)


def load_config(path: str | None = None) -> Config:
    """Defaults, overridden by `path` or by configs/default.yaml if present."""
    config = Config()
    config_path = Path(path) if path else Path("configs/default.yaml")
    if config_path.exists():
        _merge(config, yaml.safe_load(config_path.read_text()) or {})
    elif path:
        raise FileNotFoundError(path)
    return config


def dump_config(config: Config) -> str:
    return yaml.safe_dump(asdict(config), sort_keys=False)
