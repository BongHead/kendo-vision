"""Wires the stages together for one video source."""
from __future__ import annotations

import time

import cv2
import numpy as np

from .config import Config
from .constants import ESCAPE_KEY, PREVIEW_MAX_WIDTH, PREVIEW_WINDOW_TITLE
from .events.features import FeatureBuffer, Observation, raw_row
from .events.heuristic import StrikeDetector
from .geometry.targets import target_zones
from .io.audio import open_audio
from .io.sink import EventSink, VideoSink
from .io.source import FrameSource
from .match import MatchState
from .perception.player_filter import PlayerFilter
from .perception.pose import PoseEstimator
from .perception.shinai import make_shinai_estimator
from .perception.tracker import PlayerTracker
from .types import StrikeEvent
from .validity.judge import ValidityJudge
from .viz.overlay import Overlay

def _fit_to_screen(image: np.ndarray, max_width: int = PREVIEW_MAX_WIDTH) -> np.ndarray:
    height, width = image.shape[:2]
    if width <= max_width:
        return image
    return cv2.resize(image, (max_width, round(height * max_width / width)))


def run(config: Config, source: str, save: str | None = None, events: str | None = None,
        dump: str | None = None, show: bool = False) -> list[StrikeEvent]:
    """Analyse a file or camera; returns the judged strikes."""
    frame_source = FrameSource(source, config.camera)
    audio = open_audio(source, frame_source.live, config.audio, frame_source.start_time)
    pose_estimator = PoseEstimator(config.model, config.track.tracker)
    player_filter = PlayerFilter(config.track) if config.track.filter_players else None
    tracker, shinai_estimator = PlayerTracker(config.track), make_shinai_estimator(config)
    buffer, detector = FeatureBuffer(config.strike.buffer_seconds), StrikeDetector(config)
    judge, match, overlay = ValidityJudge(config, audio), MatchState(config.match), Overlay()
    video_sink = None  # opened on the first frame, once its real size is known
    event_sink = EventSink(events) if events else None
    frame_times, raw_rows, judged_events = [], [], []
    frame_count, last_time, fps = 0, 0.0, 0.0
    busy_seconds = 0.0  # processing time, not counting the warm-up frame

    def record(new_events: list[StrikeEvent], now: float) -> None:
        """Score, display, log and print each newly judged strike."""
        for event in new_events:
            match.apply(event)
            overlay.verdict(event, now)
            judged_events.append(event)
            if event_sink:
                event_sink.write(event)
            print(f"{event.time:8.2f}s  {event.attacker:5}  {event.target:5}  {event.verdict:10}  "
                  f"{event.validity:.2f}" + ("  counted" if event.counted else ""))

    try:
        for frame in frame_source:
            frame_started = time.perf_counter()
            detections = pose_estimator(frame.image)
            if player_filter is not None:
                detections = player_filter(frame.image, detections)  # kendoka only
            players = tracker.update(detections, frame)
            shinai_estimator(frame.image, players)
            players_by_side = {player.side: player for player in players}
            zones_by_side = {side: target_zones(player, config.zones) for side, player in players_by_side.items()}
            observation = Observation(frame.time, frame.index, players_by_side, zones_by_side)
            buffer.push(observation)
            for candidate in detector.step(observation):
                judge.push(candidate)
                overlay.strike(candidate)
            record(judge.poll(buffer, frame.time), frame.time)
            if dump:
                frame_times.append(frame.time)
                raw_rows.append(raw_row(players_by_side))
            frame_count, last_time = frame_count + 1, frame.time

            if save or show:
                image = overlay.draw(frame.image, observation, match, fps)
                if save:
                    if video_sink is None:  # sized from a real frame; container metadata can be wrong
                        video_sink = VideoSink(save, frame_source.fps, (image.shape[1], image.shape[0]))
                    video_sink.write(image)
                if show:
                    cv2.imshow(PREVIEW_WINDOW_TITLE, _fit_to_screen(image))
                    if cv2.waitKey(1) & 0xFF in (ord("q"), ESCAPE_KEY):
                        break

            frame_seconds = max(time.perf_counter() - frame_started, 1e-6)
            if frame_count == 1:  # the first frame includes model warm-up, so it only seeds the display
                fps = 1 / frame_seconds
            else:
                fps = 0.9 * fps + 0.1 / frame_seconds  # smoothed for the on-screen readout
                busy_seconds += frame_seconds
        # the stream is over: judge anything still waiting on its zanshin window
        record(judge.poll(buffer, last_time, final=True), last_time)
    finally:
        frame_source.release()
        for closable in (audio, video_sink, event_sink):
            if closable is not None:
                closable.close()
        if show:
            cv2.destroyAllWindows()

    if dump and raw_rows:
        arrays = {"times": np.array(frame_times), "raw": np.stack(raw_rows)}
        envelopes = audio.envelopes(0.0, last_time + 1.0) if audio is not None else None
        if envelopes is not None:
            arrays["audio_times"], arrays["audio_levels"] = envelopes
        np.savez_compressed(dump, **arrays)

    average_fps = (frame_count - 1) / max(busy_seconds, 1e-6)
    dropped = f", {frame_source.dropped} dropped" if frame_source.live else ""
    print(f"{frame_count} frames at {average_fps:.1f} fps{dropped}, {len(judged_events)} strikes, "
          f"score red {match.score['red']} - white {match.score['white']}")
    return judged_events
