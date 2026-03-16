from __future__ import annotations

import argparse
import socket
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2

from video_bandwidth.codecs import (
    AV1Encoder,
    CodecUnavailableError,
    H264Encoder,
    H265Encoder,
    VVCEncoder,
    VP9Encoder,
)
from video_bandwidth.protocol import (
    CODEC_AV1,
    CODEC_H264,
    CODEC_H265,
    CODEC_H266,
    CODEC_MJPEG,
    CODEC_VP9,
    ControlSettings,
    DEFAULT_RESOLUTION,
    RESOLUTION_PRESETS,
    SUPPORTED_RESOLUTIONS,
    ThroughputTracker,
    recv_control,
    send_frame,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read a video file or RTSP stream and forward it over TCP."
    )
    parser.add_argument(
        "--source",
        default="cars-moving-on-road.mp4",
        help="Video source. Can be a local file path or an RTSP URL.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Receiver host.")
    parser.add_argument("--port", type=int, default=5000, help="Receiver TCP port.")
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=80,
        help="Default JPEG quality until the receiver overrides it (1-100).",
    )
    parser.add_argument(
        "--resolution",
        default=DEFAULT_RESOLUTION,
        choices=list(SUPPORTED_RESOLUTIONS),
        help="Default output resolution until the receiver overrides it.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=25,
        help="Default output FPS until the receiver overrides it.",
    )
    parser.add_argument(
        "--codec",
        default=CODEC_MJPEG,
        choices=[CODEC_MJPEG, CODEC_H264, CODEC_H265, CODEC_H266, CODEC_VP9, CODEC_AV1],
        help="Default codec until the receiver overrides it.",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Restart the source when the end of a local video file is reached.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Optional frame limit for testing. 0 means no limit.",
    )
    return parser.parse_args()


def open_capture(source: str) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open source: {source}")
    return capture


def is_local_file(source: str) -> bool:
    return Path(source).exists()


def resize_frame(frame, resolution: str):
    width, height = RESOLUTION_PRESETS[resolution]
    if frame.shape[1] == width and frame.shape[0] == height:
        return frame
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


@dataclass(slots=True)
class EncoderState:
    settings: ControlSettings


def control_loop(sock: socket.socket, state: EncoderState, lock: threading.Lock) -> None:
    while True:
        try:
            settings = recv_control(sock)
        except (ConnectionError, OSError):
            return
        except Exception as exc:
            print(f"Control channel error: {exc}")
            return

        with lock:
            state.settings = settings
        print(
            "Updated by receiver | "
            f"fps={settings.target_fps} | "
            f"quality={settings.jpeg_quality} | "
            f"resolution={settings.resolution} | "
            f"codec={settings.codec}"
        )


def main() -> None:
    args = parse_args()
    capture = open_capture(args.source)

    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 1 or fps != fps:
        fps = 25.0
    local_file = is_local_file(args.source)
    frame_id = 0
    source_interval = 1.0 / fps
    next_deadline = time.monotonic()
    send_budget = 0.0
    state = EncoderState(
        settings=ControlSettings(
            target_fps=args.fps,
            jpeg_quality=args.jpeg_quality,
            resolution=args.resolution,
            codec=args.codec,
        ).normalized()
    )
    state_lock = threading.Lock()
    encoders: dict[str, H264Encoder | H265Encoder | VVCEncoder | VP9Encoder | AV1Encoder] = {}
    encoder_params: dict[str, tuple[int, int, int, int]] = {}
    codec_supported = {
        CODEC_H264: True,
        CODEC_H265: True,
        CODEC_H266: True,
        CODEC_VP9: True,
        CODEC_AV1: True,
    }
    empty_output_streak = {
        CODEC_H264: 0,
        CODEC_H265: 0,
        CODEC_H266: 0,
        CODEC_VP9: 0,
        CODEC_AV1: 0,
    }
    processed_frames = 0

    with socket.create_connection((args.host, args.port)) as sock:
        tracker = ThroughputTracker()
        control_thread = threading.Thread(
            target=control_loop,
            args=(sock, state, state_lock),
            daemon=True,
        )
        control_thread.start()
        print(f"Connected to receiver at {args.host}:{args.port}")
        print(
            f"Source={args.source} source_fps={fps:.2f} "
            f"default_fps={state.settings.target_fps} "
            f"default_quality={state.settings.jpeg_quality} "
            f"default_resolution={state.settings.resolution} "
            f"default_codec={state.settings.codec}"
        )

        while True:
            ok, frame = capture.read()
            if not ok:
                if args.loop and local_file:
                    capture.release()
                    capture = open_capture(args.source)
                    next_deadline = time.monotonic()
                    send_budget = 0.0
                    encoders = {}
                    encoder_params = {}
                    empty_output_streak = {key: 0 for key in empty_output_streak}
                    continue
                break

            with state_lock:
                settings = state.settings

            effective_fps = min(settings.target_fps, max(int(round(fps)), 1))
            send_budget += effective_fps / fps
            if send_budget >= 1.0:
                send_budget -= 1.0
                processed_frames += 1
                frame = resize_frame(frame, settings.resolution)
                codec = settings.codec
                payloads: list[bytes] = []

                if codec in (CODEC_H264, CODEC_H265, CODEC_H266, CODEC_VP9, CODEC_AV1) and codec_supported[codec]:
                    frame_params = (
                        frame.shape[1],
                        frame.shape[0],
                        effective_fps,
                        settings.jpeg_quality,
                    )
                    try:
                        if codec not in encoders or encoder_params.get(codec) != frame_params:
                            if codec == CODEC_H264:
                                encoders[codec] = H264Encoder(
                                    width=frame_params[0],
                                    height=frame_params[1],
                                    fps=frame_params[2],
                                    jpeg_quality=frame_params[3],
                                )
                            elif codec == CODEC_H265:
                                encoders[codec] = H265Encoder(
                                    width=frame_params[0],
                                    height=frame_params[1],
                                    fps=frame_params[2],
                                    jpeg_quality=frame_params[3],
                                )
                            elif codec == CODEC_H266:
                                encoders[codec] = VVCEncoder(
                                    width=frame_params[0],
                                    height=frame_params[1],
                                    fps=frame_params[2],
                                    jpeg_quality=frame_params[3],
                                )
                            elif codec == CODEC_VP9:
                                encoders[codec] = VP9Encoder(
                                    width=frame_params[0],
                                    height=frame_params[1],
                                    fps=frame_params[2],
                                    jpeg_quality=frame_params[3],
                                )
                            else:
                                encoders[codec] = AV1Encoder(
                                    width=frame_params[0],
                                    height=frame_params[1],
                                    fps=frame_params[2],
                                    jpeg_quality=frame_params[3],
                                )
                            encoder_params[codec] = frame_params
                        payloads = encoders[codec].encode_frame(frame)
                    except CodecUnavailableError as exc:
                        # Keep streaming by falling back to MJPEG if the backend is unavailable.
                        failing_codec = codec
                        print(f"{failing_codec.upper()} unavailable, fallback to MJPEG: {exc}")
                        codec_supported[failing_codec] = False
                        encoders.pop(failing_codec, None)
                        encoder_params.pop(failing_codec, None)
                        codec = CODEC_MJPEG
                elif codec in (CODEC_H264, CODEC_H265, CODEC_H266, CODEC_VP9, CODEC_AV1) and not codec_supported[codec]:
                    codec = CODEC_MJPEG

                if codec in empty_output_streak:
                    if payloads:
                        empty_output_streak[codec] = 0
                    else:
                        empty_output_streak[codec] += 1
                        if empty_output_streak[codec] >= 30:
                            print(
                                f"{codec.upper()} produced no packets for "
                                f"{empty_output_streak[codec]} frames"
                            )
                        if args.max_frames and processed_frames >= args.max_frames:
                            break
                        continue

                if codec == CODEC_MJPEG:
                    encode_params = [cv2.IMWRITE_JPEG_QUALITY, settings.jpeg_quality]
                    encoded_ok, encoded = cv2.imencode(".jpg", frame, encode_params)
                    if not encoded_ok:
                        raise RuntimeError("Unable to encode frame as JPEG")
                    payloads = [encoded.tobytes()]

                byte_count = 0
                sent_at_ns = time.time_ns()
                try:
                    for payload in payloads:
                        byte_count += send_frame(
                            sock=sock,
                            frame_id=frame_id,
                            codec=codec,
                            payload=payload,
                            sent_at_ns=sent_at_ns,
                        )
                except (ConnectionError, OSError):
                    break

                snapshot = tracker.record(byte_count, frame_count=1) if byte_count > 0 else None
                if snapshot is not None:
                    print(
                        f"TX {snapshot.elapsed_seconds:7.1f}s | "
                        f"{snapshot.bitrate_mbps:6.2f} Mb/s | "
                        f"{snapshot.fps:5.1f} fps | "
                        f"{snapshot.total_megabytes:7.2f} MB"
                    )

                frame_id += 1
                if args.max_frames and processed_frames >= args.max_frames:
                    break

            if local_file:
                next_deadline += source_interval
                sleep_for = next_deadline - time.monotonic()
                if sleep_for > 0:
                    time.sleep(sleep_for)
                else:
                    next_deadline = time.monotonic()

    capture.release()
    summary = tracker.finalize()
    print(
        f"TX summary | {summary.bitrate_mbps:.2f} Mb/s average | "
        f"{summary.fps:.1f} fps average | {summary.total_megabytes:.2f} MB sent"
    )


if __name__ == "__main__":
    main()
