"""Constants for feature extraction, strike detection and StrikeNet."""
import numpy as np

from ..constants import TARGETS

# Raw row: what was seen in one frame, in pixels. Red's block, then white's, each laid out as:
RAW_PER_SIDE = 60
RAW_SIZE = 2 * RAW_PER_SIDE
RAW_PRESENT, RAW_SCALE = 0, 1
RAW_KEYPOINTS = slice(2, 36)  # 17 x, y pairs
RAW_KEYPOINT_CONFIDENCE = slice(36, 53)
RAW_SHINAI = slice(53, 59)  # butt, tsuba, tip as x, y pairs
RAW_SHINAI_CONFIDENCE = 59

# Static features: the same, centred between the players and in torso lengths, without the scale.
FEATURES_PER_SIDE = 59
FEATURE_PRESENT = 0
FEATURE_KEYPOINTS = slice(1, 35)
FEATURE_KEYPOINT_CONFIDENCE = slice(35, 52)
FEATURE_SHINAI = slice(52, 58)
FEATURE_SHINAI_CONFIDENCE = 58
FEATURE_GAPS_START = 2 * FEATURES_PER_SIDE  # then red's gap to each of white's targets, then white's to red's
STATIC_SIZE = FEATURE_GAPS_START + 2 * len(TARGETS)
AUDIO_BAND_COUNT = 3
FEATURE_SIZE = 2 * STATIC_SIZE + AUDIO_BAND_COUNT  # static features + their velocities + audio bands
MAX_GAP = 3.0  # gaps are clipped here, and take this value when unknown

# Column groups of the static features, used for augmentation
_SIDE_X_COLUMNS = [*range(FEATURE_KEYPOINTS.start, FEATURE_KEYPOINTS.stop, 2),
                   *range(FEATURE_SHINAI.start, FEATURE_SHINAI.stop, 2)]
X_COLUMNS = np.array(_SIDE_X_COLUMNS + [FEATURES_PER_SIDE + column for column in _SIDE_X_COLUMNS])
XY_COLUMNS = np.concatenate([X_COLUMNS, X_COLUMNS + 1])
_RED_GAP_COLUMNS = FEATURE_GAPS_START + np.arange(len(TARGETS))
# column order that exchanges red and white: white's block first, and the two gap groups swapped
SWAP_SIDES = np.concatenate([np.arange(FEATURES_PER_SIDE, 2 * FEATURES_PER_SIDE), np.arange(FEATURES_PER_SIDE),
                             _RED_GAP_COLUMNS + len(TARGETS), _RED_GAP_COLUMNS])

# Strike detector
HISTORY_LENGTH = 64  # measures kept per attacker
MIN_TIP_SPEED_SPAN = 0.03  # seconds; tip speed is never measured over less than this

# StrikeNet
CLASSES = ("none", *TARGETS)
DILATIONS = (1, 2, 4, 8, 16)  # each block looks twice as far in time as the one before
