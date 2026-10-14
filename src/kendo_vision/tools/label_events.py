"""Review tool for strike events: confirm, fix or add events while scrubbing a video."""
from __future__ import annotations

import json
from pathlib import Path

import cv2

from ..constants import DEFAULT_FPS, ESCAPE_KEY, NO_KEY
from ..io.sink import read_jsonl
from ..viz.constants import FONT
from .constants import ACTIVE_WITHIN, EVENTS_HELP, EVENTS_WINDOW_TITLE, IPPON_TEXT, MAX_VIEW_WIDTH, TARGET_KEYS


def run(video: str, out: str, events: str | None = None) -> None:
    capture = cv2.VideoCapture(video)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")
    fps = capture.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS
    frame_total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    # resume from earlier labels if there are any, else start from the detector's candidates
    starting_file = out if Path(out).exists() else events
    labels = []
    for event in (read_jsonl(starting_file) if starting_file else []):
        labels.append({"time": event["time"], "frame": event["frame"], "attacker": event["attacker"],
                       "target": event["target"], "ippon": event.get("ippon")})

    def save() -> None:
        labels.sort(key=lambda label: label["time"])
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as file:
            for label in labels:
                file.write(json.dumps({"video": Path(video).stem, **label}) + "\n")
        print(f"Saved {len(labels)} events to {out}")

    def active_label(current_time: float) -> dict | None:
        """The label nearest the current frame, if it is close enough to be the one being edited."""
        nearest = min(labels, key=lambda label: abs(label["time"] - current_time), default=None)
        if nearest is not None and abs(nearest["time"] - current_time) <= ACTIVE_WITHIN:
            return nearest
        return None

    frame_index, shown_index, playing, image = 0, -1, False, None
    while True:
        if frame_index != shown_index:
            if frame_index != shown_index + 1:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)  # a jump; the next frame needs no seek
            success, new_image = capture.read()
            if success:
                image, shown_index = new_image, frame_index
            else:  # ran off the end: stay on the last frame that could be read
                frame_index, playing = max(shown_index, 0), False
        if image is None:
            raise RuntimeError("Video has no readable frames")

        current_time = frame_index / fps
        label = active_label(current_time)
        canvas = image.copy()
        height, width = canvas.shape[:2]
        if width > MAX_VIEW_WIDTH:
            canvas = cv2.resize(canvas, (MAX_VIEW_WIDTH, round(height * MAX_VIEW_WIDTH / width)))
            height, width = canvas.shape[:2]
        if label is None:
            label_text = "no event here"
        else:
            label_text = (f"event @ {label['time']:.2f}s  {label['attacker']}  {label['target']}  "
                          f"ippon: {IPPON_TEXT[label['ippon']]}")
        status = f"{current_time:7.2f}s  frame {frame_index}/{frame_total}  |  {label_text}"
        for line_index, text in enumerate((status, *EVENTS_HELP)):
            position = (8, 22 + 20 * line_index)
            cv2.putText(canvas, text, position, FONT, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(canvas, text, position, FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        # timeline along the bottom: a tick per event (brighter for the active one), a white bar for now
        for other_label in labels:
            tick_x = int(other_label["frame"] / max(frame_total, 1) * width)
            tick_color = (0, 220, 255) if other_label is label else (0, 140, 255)
            cv2.line(canvas, (tick_x, height - 14), (tick_x, height - 2), tick_color, 2)
        cursor_x = int(frame_index / max(frame_total, 1) * width)
        cv2.line(canvas, (cursor_x, height - 18), (cursor_x, height), (255, 255, 255), 1)
        cv2.imshow(EVENTS_WINDOW_TITLE, canvas)

        # while playing, wait one frame's duration so playback runs at normal speed
        wait_milliseconds = max(1, round(1000 / fps)) if playing else 30
        key = cv2.waitKey(wait_milliseconds) & 0xFF
        pressed = chr(key) if key < 128 else "\0"  # never "", which is `in` every string
        if key == NO_KEY:
            if playing:
                frame_index = min(frame_index + 1, frame_total - 1)
        elif pressed == " ":
            playing = not playing
        elif pressed in ",.[]":
            playing = False
            frames_to_move = {",": -1, ".": 1, "[": -round(fps), "]": round(fps)}[pressed]
            frame_index = min(max(frame_index + frames_to_move, 0), frame_total - 1)
        elif pressed in "np":
            if pressed == "n":  # the first event after this frame
                later = [candidate for candidate in labels if candidate["frame"] > frame_index]
                destination = min(later, key=lambda candidate: candidate["frame"], default=None)
            else:  # the last event before this frame
                earlier = [candidate for candidate in labels if candidate["frame"] < frame_index]
                destination = max(earlier, key=lambda candidate: candidate["frame"], default=None)
            if destination is not None:
                frame_index, playing = destination["frame"], False
        elif pressed in TARGET_KEYS:
            if label is None:  # nothing here yet: this key press creates the event
                label = {"time": round(current_time, 3), "frame": frame_index, "attacker": "red",
                         "target": "men", "ippon": None}
                labels.append(label)
            label["target"] = TARGET_KEYS[pressed]
        elif label is not None and pressed in "rw":
            label["attacker"] = "red" if pressed == "r" else "white"
        elif label is not None and pressed in "io":
            label["ippon"] = pressed == "i"
        elif label is not None and pressed == "x":
            labels.remove(label)
        elif pressed == "s":
            save()
        elif pressed == "q" or key == ESCAPE_KEY:
            save()
            break
    capture.release()
    cv2.destroyAllWindows()
