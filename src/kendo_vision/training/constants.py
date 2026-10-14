"""Constants for building the strike dataset and training StrikeNet."""

UNKNOWN = -1  # label value for "not labelled"; the loss skips it
NONE_CLASS = 0  # index of "none" (not a strike) among the target classes
TIME_JITTER = 0.1  # seconds a training window may be shifted off its strike
COORDINATE_NOISE = 0.02  # torso lengths of noise added to every coordinate in training
HEADS = ("target", "attacker", "ippon")  # StrikeNet's three outputs
