"""Audio on the video's clock: a file's audio track or a live microphone."""
from __future__ import annotations

import threading
import time

import numpy as np
from scipy import signal

from ..config import AudioConfig
from .constants import BANDS, CHUNK_PADDING, CHUNK_SECONDS, HOP_SECONDS, LIVE_BUFFER_SECONDS, STFT_WINDOW_SAMPLES


def band_envelopes(samples: np.ndarray, sample_rate: int):
    """Band levels in dB every 5 ms: (times, (count, 3) array in BANDS order)."""
    if len(samples) < STFT_WINDOW_SAMPLES:
        return None
    hop = int(sample_rate * HOP_SECONDS)
    frequencies, times, spectrum = signal.stft(samples, fs=sample_rate, nperseg=STFT_WINDOW_SAMPLES,
                                               noverlap=STFT_WINDOW_SAMPLES - hop, boundary=None, padded=False)
    power = np.abs(spectrum) ** 2
    levels = []
    for low_edge, high_edge in BANDS.values():
        in_band = (frequencies >= low_edge) & (frequencies < high_edge)
        levels.append(10 * np.log10(power[in_band].sum(axis=0) + 1e-12))
    return times, np.stack(levels, axis=1)


class AudioSource:
    sample_rate: int

    def samples(self, start: float, end: float):
        """(mono samples, their start time) for the part of [start, end] that is available, or None."""
        raise NotImplementedError

    def envelopes(self, start: float, end: float):
        """Band envelopes over [start, end], or None. Works in chunks to bound memory."""
        all_times, all_levels = [], []
        chunk_start = start
        while chunk_start < end:
            chunk_end = min(chunk_start + CHUNK_SECONDS, end)
            available = self.samples(chunk_start - CHUNK_PADDING, chunk_end + CHUNK_PADDING)
            result = band_envelopes(available[0], self.sample_rate) if available is not None else None
            if result is not None:
                times = result[0] + available[1]  # shift from chunk time to video time
                inside = (times >= chunk_start) & (times < chunk_end)  # drop the padding so chunks do not overlap
                all_times.append(times[inside])
                all_levels.append(result[1][inside])
            chunk_start = chunk_end
        if not all_times or not sum(len(times) for times in all_times):
            return None
        return np.concatenate(all_times), np.concatenate(all_levels)

    def close(self) -> None:
        pass


class FileAudio(AudioSource):
    def __init__(self, waveform: np.ndarray, config: AudioConfig):
        self.waveform, self.sample_rate, self.offset = waveform, config.sample_rate, config.offset_seconds

    @classmethod
    def load(cls, path: str, config: AudioConfig) -> "FileAudio | None":
        """Decode the first audio track to mono; None if the file has none."""
        import av

        with av.open(str(path)) as container:
            if not container.streams.audio:
                return None
            resampler = av.AudioResampler(format="flt", layout="mono", rate=config.sample_rate)
            chunks = []
            for audio_frame in container.decode(audio=0):
                chunks += [resampled.to_ndarray().reshape(-1) for resampled in resampler.resample(audio_frame)]
            # passing None flushes what the resampler is still holding
            chunks += [resampled.to_ndarray().reshape(-1) for resampled in resampler.resample(None)]
        return cls(np.concatenate(chunks), config) if chunks else None

    @property
    def duration(self) -> float:
        return len(self.waveform) / self.sample_rate + self.offset

    def samples(self, start, end):
        first = max(0, int((start - self.offset) * self.sample_rate))
        last = min(len(self.waveform), int((end - self.offset) * self.sample_rate))
        if last <= first:
            return None
        return self.waveform[first:last], first / self.sample_rate + self.offset


class LiveAudio(AudioSource):
    """Ring buffer fed by the microphone, timed from the same start as the live frames."""

    def __init__(self, config: AudioConfig, start_time: float, seconds: float = LIVE_BUFFER_SECONDS):
        import sounddevice

        self.sample_rate, self.offset, self.start_time = config.sample_rate, config.offset_seconds, start_time
        self.ring = np.zeros(int(seconds * self.sample_rate), np.float32)
        self.ring_end_time = 0.0  # video time of the newest sample
        self.lock = threading.Lock()
        self.stream = sounddevice.InputStream(samplerate=self.sample_rate, channels=1, dtype="float32",
                                              device=config.mic_device, callback=self._on_audio)
        self.stream.start()

    def _on_audio(self, block, sample_count, time_info, status):
        """Called by sounddevice on its own thread with each new block of samples."""
        with self.lock:
            # shift the oldest samples out and append the new block at the end
            self.ring = np.roll(self.ring, -sample_count)
            self.ring[-sample_count:] = block[:, 0]
            self.ring_end_time = time.monotonic() - self.start_time + self.offset

    def samples(self, start, end):
        with self.lock:
            ring, ring_end_time = self.ring.copy(), self.ring_end_time
        ring_start_time = ring_end_time - len(ring) / self.sample_rate
        first = max(0, int((start - ring_start_time) * self.sample_rate))
        last = min(len(ring), int((end - ring_start_time) * self.sample_rate))
        if last <= first:
            return None
        return ring[first:last], ring_start_time + first / self.sample_rate

    def close(self) -> None:
        self.stream.stop()
        self.stream.close()


def open_audio(source: str, live: bool, config: AudioConfig, start_time: float) -> AudioSource | None:
    """Audio for the source, or None (with a note) when disabled or unavailable."""
    if not config.enabled:
        return None
    try:
        audio = LiveAudio(config, start_time) if live else FileAudio.load(source, config)
    except Exception as error:  # no mic, no codec, ...
        print(f"Audio unavailable ({error}); kiai and sound cues will be skipped.")
        return None
    if audio is None:
        print("No audio track; kiai and sound cues will be skipped.")
    return audio
