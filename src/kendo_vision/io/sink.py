"""Outputs: annotated video and a JSON-lines event log."""
from __future__ import annotations

import json
from pathlib import Path

import cv2

from ..types import StrikeEvent
from .constants import VIDEO_CODEC


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


class VideoSink:
    def __init__(self, path: str, fps: float, size: tuple[int, int]):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*VIDEO_CODEC), fps, size)
        if not self.writer.isOpened():
            raise RuntimeError(f"Cannot write video: {path}")

    def write(self, image) -> None:
        self.writer.write(image)

    def close(self) -> None:
        self.writer.release()


class EventSink:
    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.file = open(path, "w", encoding="utf-8")

    def write(self, event: StrikeEvent) -> None:
        self.file.write(json.dumps(event.to_dict()) + "\n")
        self.file.flush()  # so a crash or Ctrl+C loses nothing

    def close(self) -> None:
        self.file.close()
