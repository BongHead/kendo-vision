"""Constants for the labelling and frame-extraction tools."""

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
JPEG_QUALITY = 95  # for extracted frames

# label-shinai
SHINAI_WINDOW_TITLE = "label-shinai"
SHINAI_HELP = ("click: butt, tsuba, tip | drag: move | right-click: delete | space: save+next | b: back | "
               "x: skip | u: undo | c: clear | q: quit")
POINT_COLORS = ((255, 120, 0), (0, 200, 0), (0, 0, 255))  # butt, tsuba, tip (BGR)
POINTS_PER_SHINAI = 3
GRAB_RADIUS = 12  # screen pixels within which a click picks up an existing point
MAX_VIEW_SIZE = (1600, 900)  # larger images are shrunk to fit this
VALUES_PER_LINE = 5 + 3 * POINTS_PER_SHINAI  # class, 4 box numbers, then x, y, visibility per point

# label-events
EVENTS_WINDOW_TITLE = "label-events"
EVENTS_HELP = ("space: play/pause | , . : frame | [ ] : second | n p: next/prev event | "
               "m k d t: target (adds an event if none here)",
               "0: not a strike | r w: attacker | i o: ippon yes/no | x: delete | s: save | q: save+quit")
TARGET_KEYS = {"m": "men", "k": "kote", "d": "do", "t": "tsuki", "0": "none"}
IPPON_TEXT = {None: "?", True: "yes", False: "no"}
ACTIVE_WITHIN = 0.5  # seconds; an event this close to the current frame is the one being edited
MAX_VIEW_WIDTH = 1600
