"""Constants for video and audio input and output."""

# Audio bands in Hz. low: fumikomi stomp, voice: kiai, high: shinai impact.
BANDS = {"low": (40, 300), "voice": (300, 3000), "high": (3000, 8000)}
LOW, VOICE, HIGH = 0, 1, 2  # column of each band in an envelope array
HOP_SECONDS = 0.005  # spacing of envelope samples
STFT_WINDOW_SAMPLES = 512
CHUNK_SECONDS = 20.0  # audio is analysed in pieces of this length to bound memory
CHUNK_PADDING = 0.05  # seconds read either side of a chunk so its edge windows are complete
LIVE_BUFFER_SECONDS = 12.0  # how much microphone audio is kept

CAMERA_CODEC = "MJPG"  # what lets most webcams reach 60 fps
VIDEO_CODEC = "mp4v"  # for the annotated output video
