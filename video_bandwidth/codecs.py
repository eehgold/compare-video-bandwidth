from __future__ import annotations

from fractions import Fraction
from typing import Sequence

import cv2
import numpy as np

try:
    import av
except Exception:  # pragma: no cover - optional dependency at runtime
    av = None


class CodecUnavailableError(RuntimeError):
    """Raised when an expected codec backend is not available."""


def _quality_to_crf(jpeg_quality: int) -> int:
    # quality=100 -> CRF 18 (better), quality=1 -> CRF 45 (worse)
    return int(round(45 - ((jpeg_quality - 1) * (27.0 / 99.0))))


def _build_encoder(
    codec_label: str,
    candidates: Sequence[str],
    width: int,
    height: int,
    fps: int,
    jpeg_quality: int,
    options_candidates: Sequence[dict[str, str]] | None = None,
):
    if av is None:
        raise CodecUnavailableError("PyAV is not installed")

    errors: list[str] = []
    crf = _quality_to_crf(jpeg_quality)
    if options_candidates is None:
        options_candidates = (
            {
                "preset": "veryfast",
                "tune": "zerolatency",
                "crf": str(crf),
            },
            {},
        )

    for codec_name in candidates:
        for options in options_candidates:
            try:
                context = av.CodecContext.create(codec_name, "w")
                context.width = width
                context.height = height
                context.pix_fmt = "yuv420p"
                context.time_base = Fraction(1, max(fps, 1))
                context.framerate = Fraction(max(fps, 1), 1)
                context.options = options
                context.open()
                return context
            except Exception as exc:
                suffix = f" options={options}" if options else ""
                errors.append(f"{codec_name}{suffix}: {exc}")

    raise CodecUnavailableError(
        f"Unable to initialize a {codec_label} encoder. " + " | ".join(errors)
    )


class _PyAvEncoder:
    def __init__(
        self,
        codec_label: str,
        candidates: Sequence[str],
        width: int,
        height: int,
        fps: int,
        jpeg_quality: int,
        options_candidates: Sequence[dict[str, str]] | None = None,
    ) -> None:
        self._context = _build_encoder(
            codec_label=codec_label,
            candidates=candidates,
            width=width,
            height=height,
            fps=fps,
            jpeg_quality=jpeg_quality,
            options_candidates=options_candidates,
        )

    def encode_frame(self, frame: np.ndarray) -> list[bytes]:
        video_frame = av.VideoFrame.from_ndarray(frame, format="bgr24")
        packets = self._context.encode(video_frame)
        return [bytes(packet) for packet in packets]

    def flush(self) -> list[bytes]:
        packets = self._context.encode(None)
        return [bytes(packet) for packet in packets]


class H264Encoder(_PyAvEncoder):
    def __init__(self, width: int, height: int, fps: int, jpeg_quality: int) -> None:
        super().__init__(
            codec_label="H264",
            candidates=("libx264", "h264"),
            width=width,
            height=height,
            fps=fps,
            jpeg_quality=jpeg_quality,
        )


class H265Encoder(_PyAvEncoder):
    def __init__(self, width: int, height: int, fps: int, jpeg_quality: int) -> None:
        super().__init__(
            codec_label="H265",
            candidates=("libx265", "hevc"),
            width=width,
            height=height,
            fps=fps,
            jpeg_quality=jpeg_quality,
        )


class VVCEncoder(_PyAvEncoder):
    def __init__(self, width: int, height: int, fps: int, jpeg_quality: int) -> None:
        qp = _quality_to_crf(jpeg_quality)
        super().__init__(
            codec_label="H266/VVC",
            candidates=("libvvenc", "vvc"),
            width=width,
            height=height,
            fps=fps,
            jpeg_quality=jpeg_quality,
            options_candidates=(
                {"preset": "fast", "qp": str(qp)},
                {},
            ),
        )


class VP9Encoder(_PyAvEncoder):
    def __init__(self, width: int, height: int, fps: int, jpeg_quality: int) -> None:
        crf = _quality_to_crf(jpeg_quality)
        super().__init__(
            codec_label="VP9",
            candidates=("libvpx-vp9", "vp9"),
            width=width,
            height=height,
            fps=fps,
            jpeg_quality=jpeg_quality,
            options_candidates=(
                {
                    "deadline": "realtime",
                    "cpu-used": "8",
                    "row-mt": "1",
                    "lag-in-frames": "0",
                    "crf": str(crf),
                    "b": "0",
                },
                {
                    "deadline": "realtime",
                    "lag-in-frames": "0",
                },
                {},
            ),
        )


class AV1Encoder(_PyAvEncoder):
    def __init__(self, width: int, height: int, fps: int, jpeg_quality: int) -> None:
        crf = _quality_to_crf(jpeg_quality)
        super().__init__(
            codec_label="AV1",
            candidates=("libsvtav1", "av1", "libaom-av1"),
            width=width,
            height=height,
            fps=fps,
            jpeg_quality=jpeg_quality,
            options_candidates=(
                {
                    "preset": "8",
                    "crf": str(crf),
                    "svtav1-params": "pred-struct=1:lookahead=0:scd=0:keyint=60",
                },
                {
                    "usage": "realtime",
                    "cpu-used": "8",
                    "row-mt": "1",
                    "lag-in-frames": "0",
                    "crf": str(crf),
                    "b": "0",
                },
                {
                    "preset": "8",
                    "tune": "0",
                    "crf": str(crf),
                },
                {},
            ),
        )


class _PyAvDecoder:
    def __init__(self, codec_label: str, candidates: Sequence[str]) -> None:
        if av is None:
            raise CodecUnavailableError("PyAV is not installed")
        errors: list[str] = []
        context = None
        for codec_name in candidates:
            try:
                context = av.CodecContext.create(codec_name, "r")
                break
            except Exception as exc:
                errors.append(f"{codec_name}: {exc}")
        if context is None:
            raise CodecUnavailableError(
                f"Unable to initialize a {codec_label} decoder. " + " | ".join(errors)
            )
        self._context = context


class H264Decoder(_PyAvDecoder):
    def __init__(self) -> None:
        super().__init__("H264", ("h264",))

    def decode_packet(self, payload: bytes) -> list[np.ndarray]:
        packet = av.Packet(payload)
        frames = self._context.decode(packet)
        return [frame.to_ndarray(format="bgr24") for frame in frames]


class H265Decoder(_PyAvDecoder):
    def __init__(self) -> None:
        super().__init__("H265", ("hevc",))

    def decode_packet(self, payload: bytes) -> list[np.ndarray]:
        packet = av.Packet(payload)
        frames = self._context.decode(packet)
        return [frame.to_ndarray(format="bgr24") for frame in frames]


class VVCDecoder(_PyAvDecoder):
    def __init__(self) -> None:
        super().__init__("H266/VVC", ("libvvdec", "vvc"))

    def decode_packet(self, payload: bytes) -> list[np.ndarray]:
        packet = av.Packet(payload)
        frames = self._context.decode(packet)
        return [frame.to_ndarray(format="bgr24") for frame in frames]


class VP9Decoder(_PyAvDecoder):
    def __init__(self) -> None:
        super().__init__("VP9", ("vp9",))

    def decode_packet(self, payload: bytes) -> list[np.ndarray]:
        packet = av.Packet(payload)
        frames = self._context.decode(packet)
        return [frame.to_ndarray(format="bgr24") for frame in frames]


class AV1Decoder(_PyAvDecoder):
    def __init__(self) -> None:
        super().__init__("AV1", ("libdav1d", "av1"))

    def decode_packet(self, payload: bytes) -> list[np.ndarray]:
        packet = av.Packet(payload)
        frames = self._context.decode(packet)
        return [frame.to_ndarray(format="bgr24") for frame in frames]


def decode_mjpeg(payload: bytes) -> np.ndarray | None:
    buffer = np.frombuffer(payload, dtype=np.uint8)
    return cv2.imdecode(buffer, cv2.IMREAD_COLOR)
