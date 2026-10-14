"""Constants for validity scoring."""
from ..constants import LEFT_ELBOW, LEFT_SHOULDER, LEFT_WRIST, RIGHT_ELBOW, RIGHT_SHOULDER, RIGHT_WRIST

# each arm as (shoulder, elbow, wrist)
ARMS = ((LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST), (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST))
MIN_BODY_SPEED_SPAN = 0.05  # seconds; body speed is never measured over less than this
LOWEST_SCORE = 0.05  # floor, so one zero does not wipe out the whole geometric mean
RECENT_SECONDS = 10.0  # how long strikes are remembered for spotting counters
