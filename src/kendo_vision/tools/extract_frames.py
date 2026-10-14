"""Pulls frames out of a video for labelling."""
from __future__ import annotations

from pathlib import Path

import cv2

from ..constants import DEFAULT_FPS
from ..io.sink import read_jsonl
from .constants import JPEG_QUALITY


def extract(video: str, out: str, every: float = 0.5, events: str | None = None, pad: float = 0.3) -> int:
    """Save a frame every `every` seconds, plus every frame within `pad` seconds of each event."""
    capture = cv2.VideoCapture(video)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")
    fps = capture.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS
    frames_between_saves = max(1, round(every * fps))
    # (start, end) time spans around each event, where every frame is kept
    event_spans = [(event["time"] - pad, event["time"] + pad) for event in read_jsonl(events)] if events else []
    output_folder, video_name = Path(out), Path(video).stem
    output_folder.mkdir(parents=True, exist_ok=True)
    frame_index = saved_count = 0
    while True:
        success, image = capture.read()
        if not success:
            break
        frame_time = frame_index / fps
        on_schedule = frame_index % frames_between_saves == 0
        near_event = any(start <= frame_time <= end for start, end in event_spans)
        if on_schedule or near_event:
            image_path = output_folder / f"{video_name}_{frame_index:06d}.jpg"
            cv2.imwrite(str(image_path), image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            saved_count += 1
        frame_index += 1
    capture.release()
    print(f"Saved {saved_count} of {frame_index} frames to {output_folder}")
    return saved_count
