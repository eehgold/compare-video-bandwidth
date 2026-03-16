from __future__ import annotations

import socket
import struct
import time
import json
from dataclasses import dataclass

MAGIC = b"VBW1"
HEADER_STRUCT = struct.Struct("!4sQQIB")
HEADER_SIZE = HEADER_STRUCT.size
CONTROL_MAGIC = b"CTL1"
CONTROL_HEADER_STRUCT = struct.Struct("!4sI")
CONTROL_HEADER_SIZE = CONTROL_HEADER_STRUCT.size
CODEC_MJPEG = "mjpeg"
CODEC_H264 = "h264"
CODEC_H265 = "h265"
CODEC_VP9 = "vp9"
CODEC_AV1 = "av1"
SUPPORTED_CODECS = (CODEC_MJPEG, CODEC_H264, CODEC_H265, CODEC_VP9, CODEC_AV1)
CODEC_TO_ID = {CODEC_MJPEG: 0, CODEC_H264: 1, CODEC_H265: 2, CODEC_VP9: 3, CODEC_AV1: 4}
ID_TO_CODEC = {value: key for key, value in CODEC_TO_ID.items()}
RESOLUTION_360P = "640x360"
RESOLUTION_480P = "854x480"
RESOLUTION_720P = "1280x720"
RESOLUTION_1080P = "1920x1080"
RESOLUTION_PRESETS = {
    RESOLUTION_360P: (640, 360),
    RESOLUTION_480P: (854, 480),
    RESOLUTION_720P: (1280, 720),
    RESOLUTION_1080P: (1920, 1080),
}
SUPPORTED_RESOLUTIONS = tuple(RESOLUTION_PRESETS.keys())
DEFAULT_RESOLUTION = RESOLUTION_720P


class ProtocolError(RuntimeError):
    """Raised when the sender/receiver protocol is invalid."""


@dataclass(slots=True)
class ControlSettings:
    target_fps: int
    jpeg_quality: int
    resolution: str = DEFAULT_RESOLUTION
    codec: str = CODEC_MJPEG

    def normalized(self) -> "ControlSettings":
        codec = self.codec.lower()
        if codec not in SUPPORTED_CODECS:
            codec = CODEC_MJPEG
        resolution = self.resolution
        if resolution not in SUPPORTED_RESOLUTIONS:
            resolution = DEFAULT_RESOLUTION
        return ControlSettings(
            target_fps=max(1, min(self.target_fps, 120)),
            jpeg_quality=max(1, min(self.jpeg_quality, 100)),
            resolution=resolution,
            codec=codec,
        )


def send_frame(
    sock: socket.socket,
    frame_id: int,
    codec: str,
    payload: bytes,
    sent_at_ns: int,
) -> int:
    codec_id = CODEC_TO_ID.get(codec, CODEC_TO_ID[CODEC_MJPEG])
    header = HEADER_STRUCT.pack(MAGIC, frame_id, sent_at_ns, len(payload), codec_id)
    sock.sendall(header)
    sock.sendall(payload)
    return HEADER_SIZE + len(payload)


def recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = sock.recv(size - len(chunks))
        if not chunk:
            raise ConnectionError("Socket closed while receiving data")
        chunks.extend(chunk)
    return bytes(chunks)


def recv_frame(sock: socket.socket) -> tuple[int, int, str, bytes, int]:
    header = recv_exact(sock, HEADER_SIZE)
    magic, frame_id, sent_at_ns, payload_size, codec_id = HEADER_STRUCT.unpack(header)
    if magic != MAGIC:
        raise ProtocolError("Invalid protocol magic")
    codec = ID_TO_CODEC.get(codec_id)
    if codec is None:
        raise ProtocolError(f"Unsupported codec id: {codec_id}")
    payload = recv_exact(sock, payload_size)
    return frame_id, sent_at_ns, codec, payload, HEADER_SIZE + payload_size


def send_control(sock: socket.socket, settings: ControlSettings) -> int:
    normalized = settings.normalized()
    payload = json.dumps(
        {
            "target_fps": normalized.target_fps,
            "jpeg_quality": normalized.jpeg_quality,
            "resolution": normalized.resolution,
            "codec": normalized.codec,
        }
    ).encode("utf-8")
    header = CONTROL_HEADER_STRUCT.pack(CONTROL_MAGIC, len(payload))
    sock.sendall(header)
    sock.sendall(payload)
    return CONTROL_HEADER_SIZE + len(payload)


def recv_control(sock: socket.socket) -> ControlSettings:
    header = recv_exact(sock, CONTROL_HEADER_SIZE)
    magic, payload_size = CONTROL_HEADER_STRUCT.unpack(header)
    if magic != CONTROL_MAGIC:
        raise ProtocolError("Invalid control protocol magic")
    payload = recv_exact(sock, payload_size)
    data = json.loads(payload.decode("utf-8"))
    return ControlSettings(
        target_fps=int(data["target_fps"]),
        jpeg_quality=int(data["jpeg_quality"]),
        resolution=str(data.get("resolution", DEFAULT_RESOLUTION)),
        codec=str(data.get("codec", CODEC_MJPEG)),
    ).normalized()


@dataclass(slots=True)
class ThroughputSnapshot:
    bitrate_mbps: float
    fps: float
    total_megabytes: float
    elapsed_seconds: float


class ThroughputTracker:
    def __init__(self) -> None:
        now = time.monotonic()
        self._started_at = now
        self._window_started_at = now
        self._window_bytes = 0
        self._window_frames = 0
        self._total_bytes = 0
        self._total_frames = 0

    def record(self, byte_count: int, frame_count: int = 1) -> ThroughputSnapshot | None:
        self._window_bytes += byte_count
        self._window_frames += frame_count
        self._total_bytes += byte_count
        self._total_frames += frame_count

        now = time.monotonic()
        elapsed = now - self._window_started_at
        if elapsed < 1.0:
            return None

        snapshot = ThroughputSnapshot(
            bitrate_mbps=(self._window_bytes * 8) / elapsed / 1_000_000,
            fps=self._window_frames / elapsed,
            total_megabytes=self._total_bytes / 1_000_000,
            elapsed_seconds=now - self._started_at,
        )
        self._window_started_at = now
        self._window_bytes = 0
        self._window_frames = 0
        return snapshot

    def finalize(self) -> ThroughputSnapshot:
        now = time.monotonic()
        elapsed = max(now - self._started_at, 1e-9)
        return ThroughputSnapshot(
            bitrate_mbps=(self._total_bytes * 8) / elapsed / 1_000_000,
            fps=self._total_frames / elapsed,
            total_megabytes=self._total_bytes / 1_000_000,
            elapsed_seconds=elapsed,
        )
