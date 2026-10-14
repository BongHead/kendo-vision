"""Constants shared across the whole package."""

# COCO keypoint indices, as produced by the pose model
NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR = 0, 1, 2, 3, 4
LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW, LEFT_WRIST, RIGHT_WRIST = 5, 6, 7, 8, 9, 10
LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE = 11, 12, 13, 14, 15, 16
FACE_KEYPOINTS = (NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR)
KEYPOINT_COUNT = 17
MIN_KEYPOINT_CONFIDENCE = 0.3  # keypoints below this count as missing

SIDES = ("red", "white")
TARGETS = ("men", "kote", "do", "tsuki")

DEFAULT_FPS = 30.0  # used when a camera or file does not report a usable frame rate

# cv2.waitKey codes
ENTER_KEY, ESCAPE_KEY = 13, 27
NO_KEY = 255  # returned when no key was pressed

# preview window of `run --show`
PREVIEW_WINDOW_TITLE = "kendo-vision (q to quit)"
PREVIEW_MAX_WIDTH = 1280  # wider frames are shrunk to this for display
