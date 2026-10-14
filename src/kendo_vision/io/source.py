"""Frames from a video file or a live camera."""
from __future__ import annotations

import math
import sys
import threading
import time

import cv2

from ..config import CameraConfig
from ..constants import DEFAULT_FPS
from ..types import Frame
from .constants import CAMERA_CODEC


class FrameSource:
    """Iterates frames. A numeric source is a camera index; anything else is a file path."""

    def __init__(self, source: str, camera: CameraConfig):
        self.live = str(source).isdigit()
        if self.live:
            backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
            self.capture = cv2.VideoCapture(int(source), backend)
            self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*CAMERA_CODEC))
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, camera.width)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, camera.height)
            self.capture.set(cv2.CAP_PROP_FPS, camera.fps)
        else:
            self.capture = cv2.VideoCapture(str(source))
        if not self.capture.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")
        reported_fps = self.capture.get(cv2.CAP_PROP_FPS)
        # some cameras and files report 0 or NaN
        self.fps = reported_fps if math.isfinite(reported_fps) and reported_fps > 1 else DEFAULT_FPS
        self.size = (int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                     int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.dropped = 0
        self.start_time = time.monotonic()  # live timestamps are measured from here

    def __iter__(self):
        return self._live_frames() if self.live else self._file_frames()

    def _file_frames(self):
        index = 0
        while True:
            success, image = self.capture.read()
            if not success:
                break
            yield Frame(index, index / self.fps, image)
            index += 1

    def _live_frames(self):
        """A reader thread keeps only the newest frame, so a slow consumer drops frames instead of lagging."""
        lock, stop = threading.Lock(), threading.Event()
        newest: list[Frame | None] = [None]  # one-slot mailbox shared with the reader thread

        def read_camera():
            index = 0
            while not stop.is_set():
                success, image = self.capture.read()
                if not success:
                    break
                with lock:
                    if newest[0] is not None:  # the consumer never picked up the previous frame
                        self.dropped += 1
                    newest[0] = Frame(index, time.monotonic() - self.start_time, image)
                index += 1
            stop.set()

        reader = threading.Thread(target=read_camera, daemon=True)
        reader.start()
        try:
            while True:
                with lock:  # take the frame and empty the mailbox in one step
                    frame, newest[0] = newest[0], None
                if frame is not None:
                    yield frame
                elif stop.is_set():
                    break
                else:
                    time.sleep(0.001)  # nothing new yet
        finally:
            stop.set()
            reader.join(timeout=1.0)

    def release(self) -> None:
        self.capture.release()
