"""Constants for pose, tracking and shinai estimation."""
from ..constants import LEFT_HIP, LEFT_SHOULDER, RIGHT_HIP, RIGHT_SHOULDER

TORSO_KEYPOINTS = [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP]
TSUBA_POSITION = 0.25  # how far along the shinai, from the butt, the tsuba sits
TSUKA_TO_BLADE = 1 / 3  # handle length relative to the blade (tsuba to tip)
MAX_SHINAI_MATCH_COST = 1.0  # a detected shinai further than this from a player's hands is not theirs
MAX_GRIP_DISTANCE = 0.9  # torso lengths between a found blade's near end and the player's hands
MIN_BLADE_ELONGATION = 4.0  # length / width of a mask for it to count as a blade
EXPECTED_BLADE_LENGTH = 1.7  # torso lengths from tsuba to tip (about 90 cm against a 50 cm torso)
MAX_BLADE_LENGTH = 2.3  # a mask longer than this, in torso lengths, is two crossed shinai merged

# Zero-shot person classifier (OpenCLIP). Several descriptions per class are averaged.
CLIP_MODEL = "ViT-B-32"
CLIP_WEIGHTS = "laion2b_s34b_b79k"
PERSON_DESCRIPTIONS = {  # the first class is the one kept
    "kendoka": (
        "a kendo fencer wearing a men helmet with a metal face grille",
        "a kendo player in dark armor and a long dark hakama skirt, barefoot",
        "a person in kendo bogu armor holding a bamboo sword",
    ),
    "shinpan": (
        "a man in a dark business suit",
        "a man in a suit seen from behind",
        "a referee in a suit and tie holding red and white flags",
        "a man in a suit and trousers walking, partly cut off",
    ),
    "spectator": (
        "a seated spectator or photographer with a camera",
        "a person sitting on the floor",
    ),
}
CLIP_LOGIT_SCALE = 100.0  # CLIP's usual temperature for turning similarities into probabilities
MIN_CROP_PIXELS = 10  # people smaller than this are not classified

# Text prompts for the open-vocabulary shinai detector (YOLOE)
SHINAI_PROMPTS = ("kendo shinai", "bamboo sword")
