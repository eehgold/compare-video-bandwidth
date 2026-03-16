from __future__ import annotations

import argparse
import socket
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

from video_bandwidth.codecs import (
    AV1Decoder,
    CodecUnavailableError,
    H264Decoder,
    H265Decoder,
    VVCDecoder,
    VP9Decoder,
    decode_mjpeg,
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
    SUPPORTED_RESOLUTIONS,
    ThroughputSnapshot,
    ThroughputTracker,
    recv_frame,
    send_control,
)
from video_bandwidth.vehicle_counter import CarCounter, CounterResult, CounterUnavailableError

WINDOW_NAME = "Video Bandwidth Receiver"
FPS_MIN = 1
FPS_MAX = 60
QUALITY_MIN = 1
QUALITY_MAX = 100
CODEC_VALUES = (CODEC_MJPEG, CODEC_H264, CODEC_H265, CODEC_H266, CODEC_VP9, CODEC_AV1)
RESOLUTION_VALUES = tuple(SUPPORTED_RESOLUTIONS)
PANEL_WIDTH = 360
PANEL_MIN_HEIGHT = 260


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Receive a TCP video stream, display it, and measure throughput."
    )
    parser.add_argument("--bind", default="0.0.0.0", help="Bind address.")
    parser.add_argument("--port", type=int, default=5000, help="TCP port to listen on.")
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable the real-time window. Useful for headless tests.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Optional frame limit for testing. 0 means no limit.",
    )
    parser.add_argument(
        "--control-fps",
        type=int,
        default=25,
        help="Initial FPS requested by the receiver.",
    )
    parser.add_argument(
        "--control-quality",
        type=int,
        default=80,
        help="Initial JPEG quality requested by the receiver.",
    )
    parser.add_argument(
        "--control-resolution",
        default=DEFAULT_RESOLUTION,
        choices=list(RESOLUTION_VALUES),
        help="Initial output resolution requested by the receiver.",
    )
    parser.add_argument(
        "--control-codec",
        default=CODEC_MJPEG,
        choices=list(CODEC_VALUES),
        help="Initial codec requested by the receiver.",
    )
    parser.add_argument(
        "--enable-car-counter",
        action="store_true",
        help="Enable car counting by default (YOLO) on receiver side.",
    )
    parser.add_argument(
        "--yolo-model",
        default="yolov8n.pt",
        help="YOLO model file or model name used for car detection.",
    )
    return parser.parse_args()


@dataclass(slots=True)
class ReceiverUiState:
    settings: ControlSettings
    last_sent: ControlSettings | None = None
    active_codec: str = CODEC_MJPEG


class ControlsUi:
    def __init__(
        self,
        initial_settings: ControlSettings,
        initial_car_counter_enabled: bool,
    ) -> None:
        self._settings = initial_settings.normalized()
        self._car_counter_enabled = initial_car_counter_enabled
        self._force_car_counter_enabled: bool | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            args=(self._settings, initial_car_counter_enabled),
            daemon=True,
        )
        self._thread.start()
        self._ready.wait(timeout=2.0)

    def _run(self, initial_settings: ControlSettings, initial_car_counter_enabled: bool) -> None:
        try:
            import tkinter as tk
            from tkinter import ttk
        except Exception as exc:
            print(f"Controls UI unavailable: {exc}")
            self._ready.set()
            return

        root = tk.Tk()
        root.title("Receiver Controls")
        root.geometry("340x620")
        root.resizable(False, False)

        container = ttk.Frame(root, padding=12)
        container.pack(fill=tk.BOTH, expand=True)

        fps_var = tk.IntVar(value=initial_settings.target_fps)
        quality_var = tk.IntVar(value=initial_settings.jpeg_quality)
        codec_var = tk.StringVar(
            value=initial_settings.codec
            if initial_settings.codec in CODEC_VALUES
            else CODEC_VALUES[0]
        )
        resolution_var = tk.StringVar(
            value=initial_settings.resolution
            if initial_settings.resolution in RESOLUTION_VALUES
            else RESOLUTION_VALUES[0]
        )
        car_counter_var = tk.BooleanVar(value=initial_car_counter_enabled)

        def publish_settings() -> None:
            settings = ControlSettings(
                target_fps=fps_var.get(),
                jpeg_quality=quality_var.get(),
                resolution=resolution_var.get(),
                codec=codec_var.get(),
            ).normalized()
            with self._lock:
                self._settings = settings
                self._car_counter_enabled = bool(car_counter_var.get())

        def on_change(*_: object) -> None:
            publish_settings()

        legend_kwargs = {
            "anchor": "w",
            "justify": "left",
            "fg": "#666666",
            "wraplength": 305,
        }

        ttk.Label(container, text="Codec").pack(anchor="w")
        codec_combo = ttk.Combobox(
            container,
            state="readonly",
            textvariable=codec_var,
            values=list(CODEC_VALUES),
        )
        codec_combo.pack(fill="x")
        tk.Label(
            container,
            text="Type de compression utilise pour encoder le flux envoye.",
            **legend_kwargs,
        ).pack(fill="x", pady=(2, 10))

        ttk.Label(container, text="Resolution").pack(anchor="w")
        resolution_combo = ttk.Combobox(
            container,
            state="readonly",
            textvariable=resolution_var,
            values=list(RESOLUTION_VALUES),
        )
        resolution_combo.pack(fill="x")
        tk.Label(
            container,
            text="Taille des images encodees. Plus grand = plus de details et plus de debit.",
            **legend_kwargs,
        ).pack(fill="x", pady=(2, 10))

        ttk.Label(container, text="FPS").pack(anchor="w")
        fps_scale = tk.Scale(
            container,
            from_=FPS_MIN,
            to=FPS_MAX,
            orient=tk.HORIZONTAL,
            variable=fps_var,
            command=on_change,
        )
        fps_scale.pack(fill="x")
        tk.Label(
            container,
            text="Nombre d'images par seconde. Plus haut = mouvement plus fluide et plus de debit.",
            **legend_kwargs,
        ).pack(fill="x", pady=(2, 10))

        ttk.Label(container, text="Quality").pack(anchor="w")
        quality_scale = tk.Scale(
            container,
            from_=QUALITY_MIN,
            to=QUALITY_MAX,
            orient=tk.HORIZONTAL,
            variable=quality_var,
            command=on_change,
        )
        quality_scale.pack(fill="x")
        tk.Label(
            container,
            text="Qualite d'encodage. Plus haut = meilleure image, mais plus de debit.",
            **legend_kwargs,
        ).pack(fill="x", pady=(2, 8))

        car_counter_check = tk.Checkbutton(
            container,
            text="Activer compteur de voitures (YOLO)",
            variable=car_counter_var,
            onvalue=True,
            offvalue=False,
            command=on_change,
            anchor="w",
            justify="left",
        )
        car_counter_check.pack(fill="x", pady=(4, 0))
        tk.Label(
            container,
            text=(
                "Detecte les voitures, affiche les boites et compte les vehicules "
                "qui traversent la ligne en bas de l'image."
            ),
            **legend_kwargs,
        ).pack(fill="x", pady=(2, 8))

        hint = ttk.Label(container, text="Changes are applied immediately")
        hint.pack(anchor="w", pady=(10, 0))

        codec_combo.bind("<<ComboboxSelected>>", on_change)
        resolution_combo.bind("<<ComboboxSelected>>", on_change)
        publish_settings()
        self._ready.set()

        while not self._stop.is_set():
            forced_counter_state: bool | None = None
            with self._lock:
                forced_counter_state = self._force_car_counter_enabled
                self._force_car_counter_enabled = None
            if (
                forced_counter_state is not None
                and bool(car_counter_var.get()) != forced_counter_state
            ):
                car_counter_var.set(forced_counter_state)
                publish_settings()
            try:
                root.update_idletasks()
                root.update()
            except tk.TclError:
                break
            time.sleep(0.02)

        try:
            root.destroy()
        except Exception:
            pass

    def get_settings(self) -> ControlSettings:
        with self._lock:
            return self._settings

    def is_car_counter_enabled(self) -> bool:
        with self._lock:
            return self._car_counter_enabled

    def set_car_counter_enabled(self, enabled: bool) -> None:
        with self._lock:
            self._force_car_counter_enabled = bool(enabled)

    def close(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)


def maybe_send_controls(
    conn: socket.socket,
    ui_state: ReceiverUiState,
    display_enabled: bool,
    controls_ui: ControlsUi | None,
) -> bool:
    settings = controls_ui.get_settings() if (display_enabled and controls_ui is not None) else ui_state.settings
    if ui_state.last_sent == settings:
        return True
    try:
        send_control(conn, settings)
    except (ConnectionError, OSError):
        return False
    ui_state.settings = settings
    ui_state.last_sent = settings
    print(
        "Control sent to sender | "
        f"fps={settings.target_fps} | "
        f"quality={settings.jpeg_quality} | "
        f"resolution={settings.resolution} | "
        f"codec={settings.codec}"
    )
    return True


def format_stats(snapshot: ThroughputSnapshot | None) -> tuple[str, str, str]:
    if snapshot is None:
        return ("Waiting for stats...", "-- Mb/s", "-- fps")
    return (
        f"{snapshot.bitrate_mbps:.2f} Mb/s",
        f"{snapshot.fps:.1f} fps",
        f"{snapshot.total_megabytes:.2f} MB",
    )


def build_display_frame(
    frame: np.ndarray,
    snapshot: ThroughputSnapshot | None,
    settings: ControlSettings,
    active_codec: str,
    latency_ms: float,
    jitter_ms: float,
    drop_frames: int,
    drop_rate_percent: float,
    car_counter_enabled: bool,
    counted_cars: int,
    detected_cars: int,
    line_y: int | None,
    car_counter_status: str,
) -> np.ndarray:
    bitrate_text, fps_text, total_text = format_stats(snapshot)
    canvas_height = max(frame.shape[0], PANEL_MIN_HEIGHT)
    video_canvas = np.zeros((canvas_height, frame.shape[1], 3), dtype=np.uint8)
    video_canvas[: frame.shape[0], : frame.shape[1]] = frame

    panel = np.full((canvas_height, PANEL_WIDTH, 3), (28, 28, 28), dtype=np.uint8)
    cv2.rectangle(panel, (0, 0), (PANEL_WIDTH - 1, canvas_height - 1), (60, 60, 60), 1)

    lines = [
        ("Reception", (240, 240, 240), 0.9),
        (bitrate_text, (80, 220, 120), 0.8),
        (fps_text, (80, 220, 120), 0.8),
        (total_text, (80, 220, 120), 0.8),
        (f"Latence: {latency_ms:.1f} ms", (80, 220, 120), 0.75),
        (f"Jitter: {jitter_ms:.1f} ms", (80, 220, 120), 0.75),
        (f"Drops: {drop_frames} ({drop_rate_percent:.2f}%)", (80, 220, 120), 0.75),
        (
            f"Compteur voitures: {'ON' if car_counter_enabled else 'OFF'}",
            (80, 220, 120),
            0.75,
        ),
        (f"Voitures comptees: {counted_cars}", (80, 220, 120), 0.75),
        (f"Detections YOLO: {detected_cars}", (80, 220, 120), 0.75),
        (
            f"Ligne de comptage: y={line_y}" if line_y is not None else "Ligne de comptage: --",
            (80, 220, 120),
            0.75,
        ),
        (f"Statut YOLO: {car_counter_status}", (80, 220, 120), 0.7),
        ("", (0, 0, 0), 0.0),
        ("Controle emetteur", (240, 240, 240), 0.9),
        (f"FPS cible: {settings.target_fps}", (0, 215, 255), 0.75),
        (f"Qualite JPEG: {settings.jpeg_quality}", (0, 215, 255), 0.75),
        (f"Resolution: {settings.resolution}", (0, 215, 255), 0.75),
        (f"Codec demande: {settings.codec.upper()}", (0, 215, 255), 0.75),
        ("", (0, 0, 0), 0.0),
        ("Flux recu", (240, 240, 240), 0.9),
        (f"Codec recu: {active_codec.upper()}", (210, 210, 210), 0.75),
        (f"Frame: {frame.shape[1]}x{frame.shape[0]}", (210, 210, 210), 0.75),
        ("Quitter: touche q", (210, 210, 210), 0.75),
    ]

    y = 36
    for text, color, scale in lines:
        if not text:
            y += 18
            continue
        cv2.putText(
            panel,
            text,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            2,
            cv2.LINE_AA,
        )
        y += 34

    return np.hstack((video_canvas, panel))


def main() -> None:
    args = parse_args()
    last_snapshot: ThroughputSnapshot | None = None
    received_frames = 0
    ui_state = ReceiverUiState(
        settings=ControlSettings(
            target_fps=args.control_fps,
            jpeg_quality=args.control_quality,
            resolution=args.control_resolution,
            codec=args.control_codec,
        ).normalized()
    )
    h264_decoder: H264Decoder | None = None
    h264_decoder_available = True
    h265_decoder: H265Decoder | None = None
    h265_decoder_available = True
    h266_decoder: VVCDecoder | None = None
    h266_decoder_available = True
    vp9_decoder: VP9Decoder | None = None
    vp9_decoder_available = True
    av1_decoder: AV1Decoder | None = None
    av1_decoder_available = True
    controls_ui: ControlsUi | None = None
    car_counter: CarCounter | None = None

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((args.bind, args.port))
        server.listen(1)
        print(f"Receiver listening on {args.bind}:{args.port}")

        conn, address = server.accept()
        with conn:
            tracker = ThroughputTracker()
            metrics_codec: str | None = None
            last_frame_id: int | None = None
            received_unique_frames = 0
            lost_frames = 0
            decoded_frames_total = 0
            latency_samples = 0
            latency_avg_ms = 0.0
            latency_jitter_ms = 0.0
            last_latency_ms: float | None = None
            drop_rate_percent = 0.0
            car_counter_enabled = args.enable_car_counter and not args.no_display
            counter_ui_prev = car_counter_enabled
            counted_cars = 0
            detected_cars = 0
            line_y: int | None = None
            car_counter_status = "desactive"
            print(f"Accepted connection from {address[0]}:{address[1]}")
            if not args.no_display:
                controls_ui = ControlsUi(
                    initial_settings=ui_state.settings,
                    initial_car_counter_enabled=car_counter_enabled,
                )

            def enable_car_counter() -> bool:
                nonlocal car_counter, counted_cars, detected_cars, line_y, car_counter_status
                try:
                    if car_counter is None:
                        car_counter = CarCounter(model_name=args.yolo_model)
                    car_counter.reset()
                    counted_cars = 0
                    detected_cars = 0
                    line_y = None
                    car_counter_status = f"actif ({args.yolo_model})"
                    return True
                except CounterUnavailableError as exc:
                    car_counter_status = f"indisponible ({exc})"
                    print(f"Car counter unavailable: {exc}")
                    return False

            if car_counter_enabled:
                car_counter_enabled = enable_car_counter()
                if car_counter_enabled:
                    print("Car counter enabled")
                elif controls_ui is not None:
                    controls_ui.set_car_counter_enabled(False)
                    counter_ui_prev = False
            if not maybe_send_controls(
                conn,
                ui_state,
                display_enabled=not args.no_display,
                controls_ui=controls_ui,
            ):
                pass
            else:
                while True:
                    try:
                        if not maybe_send_controls(
                            conn,
                            ui_state,
                            display_enabled=not args.no_display,
                            controls_ui=controls_ui,
                        ):
                            break
                        if not args.no_display and controls_ui is not None:
                            desired_counter_enabled = controls_ui.is_car_counter_enabled()
                            if desired_counter_enabled != counter_ui_prev:
                                counter_ui_prev = desired_counter_enabled
                                if desired_counter_enabled:
                                    car_counter_enabled = enable_car_counter()
                                    if car_counter_enabled:
                                        print("Car counter enabled")
                                    elif controls_ui is not None:
                                        controls_ui.set_car_counter_enabled(False)
                                        counter_ui_prev = False
                                else:
                                    car_counter_enabled = False
                                    counted_cars = 0
                                    detected_cars = 0
                                    line_y = None
                                    car_counter_status = "desactive"
                                    print("Car counter disabled")
                        frame_id, sent_at_ns, codec, payload, byte_count = recv_frame(conn)
                    except ConnectionError:
                        break

                    if metrics_codec != codec:
                        tracker = ThroughputTracker()
                        last_snapshot = None
                        metrics_codec = codec
                        last_frame_id = None
                        received_unique_frames = 0
                        lost_frames = 0
                        decoded_frames_total = 0
                        latency_samples = 0
                        latency_avg_ms = 0.0
                        latency_jitter_ms = 0.0
                        last_latency_ms = None
                        if car_counter_enabled and car_counter is not None:
                            car_counter.reset()
                        counted_cars = 0
                        detected_cars = 0
                        line_y = None
                        if car_counter_enabled:
                            car_counter_status = f"actif ({args.yolo_model})"
                        else:
                            car_counter_status = "desactive"
                        print(f"Codec switched to {codec.upper()} -> indicators reset")

                    if last_frame_id is None:
                        last_frame_id = frame_id
                        received_unique_frames = 1
                    elif frame_id > last_frame_id:
                        lost_frames += max(frame_id - last_frame_id - 1, 0)
                        received_unique_frames += 1
                        last_frame_id = frame_id
                    elif frame_id < last_frame_id:
                        # Unexpected restart of frame numbering on the same connection.
                        last_frame_id = frame_id
                        received_unique_frames += 1

                    decoded_frames: list[np.ndarray] = []
                    if codec == CODEC_MJPEG:
                        frame = decode_mjpeg(payload)
                        if frame is not None:
                            decoded_frames = [frame]
                    elif codec == CODEC_H264:
                        if h264_decoder_available and h264_decoder is None:
                            try:
                                h264_decoder = H264Decoder()
                            except CodecUnavailableError as exc:
                                h264_decoder_available = False
                                print(f"H264 decoder unavailable: {exc}")
                        if h264_decoder is not None:
                            try:
                                decoded_frames = h264_decoder.decode_packet(payload)
                            except Exception as exc:
                                print(f"H264 decode error: {exc}")
                                decoded_frames = []
                    elif codec == CODEC_H265:
                        if h265_decoder_available and h265_decoder is None:
                            try:
                                h265_decoder = H265Decoder()
                            except CodecUnavailableError as exc:
                                h265_decoder_available = False
                                print(f"H265 decoder unavailable: {exc}")
                        if h265_decoder is not None:
                            try:
                                decoded_frames = h265_decoder.decode_packet(payload)
                            except Exception as exc:
                                print(f"H265 decode error: {exc}")
                                decoded_frames = []
                    elif codec == CODEC_H266:
                        if h266_decoder_available and h266_decoder is None:
                            try:
                                h266_decoder = VVCDecoder()
                            except CodecUnavailableError as exc:
                                h266_decoder_available = False
                                print(f"H266/VVC decoder unavailable: {exc}")
                        if h266_decoder is not None:
                            try:
                                decoded_frames = h266_decoder.decode_packet(payload)
                            except Exception as exc:
                                print(f"H266/VVC decode error: {exc}")
                                decoded_frames = []
                    elif codec == CODEC_VP9:
                        if vp9_decoder_available and vp9_decoder is None:
                            try:
                                vp9_decoder = VP9Decoder()
                            except CodecUnavailableError as exc:
                                vp9_decoder_available = False
                                print(f"VP9 decoder unavailable: {exc}")
                        if vp9_decoder is not None:
                            try:
                                decoded_frames = vp9_decoder.decode_packet(payload)
                            except Exception as exc:
                                print(f"VP9 decode error: {exc}")
                                decoded_frames = []
                    elif codec == CODEC_AV1:
                        if av1_decoder_available and av1_decoder is None:
                            try:
                                av1_decoder = AV1Decoder()
                            except CodecUnavailableError as exc:
                                av1_decoder_available = False
                                print(f"AV1 decoder unavailable: {exc}")
                        if av1_decoder is not None:
                            try:
                                decoded_frames = av1_decoder.decode_packet(payload)
                            except Exception as exc:
                                print(f"AV1 decode error: {exc}")
                                decoded_frames = []

                    if decoded_frames:
                        now_ns = time.time_ns()
                        latency_ms = max((now_ns - sent_at_ns) / 1_000_000.0, 0.0)
                        for _ in range(len(decoded_frames)):
                            latency_samples += 1
                            latency_avg_ms += (latency_ms - latency_avg_ms) / latency_samples
                            if last_latency_ms is not None:
                                delta = abs(latency_ms - last_latency_ms)
                                latency_jitter_ms += (delta - latency_jitter_ms) / 16.0
                            last_latency_ms = latency_ms
                        decoded_frames_total += len(decoded_frames)

                    sent_frame_estimate = received_unique_frames + lost_frames
                    decode_deficit = max(received_unique_frames - decoded_frames_total, 0)
                    drop_frames = lost_frames + decode_deficit
                    drop_rate_percent = (
                        (drop_frames * 100.0 / sent_frame_estimate)
                        if sent_frame_estimate > 0
                        else 0.0
                    )

                    ui_state.active_codec = codec
                    snapshot = tracker.record(byte_count, frame_count=len(decoded_frames))
                    if snapshot is not None:
                        last_snapshot = snapshot
                        print(
                            f"RX {snapshot.elapsed_seconds:7.1f}s | "
                            f"{snapshot.bitrate_mbps:6.2f} Mb/s | "
                            f"{snapshot.fps:5.1f} fps | "
                            f"{snapshot.total_megabytes:7.2f} MB | "
                            f"lat={latency_avg_ms:6.1f} ms | "
                            f"jit={latency_jitter_ms:5.1f} ms | "
                            f"drop={drop_rate_percent:5.2f}%"
                        )
                    stop_requested = False
                    for frame in decoded_frames:
                        frame_for_display = frame
                        if car_counter_enabled and car_counter is not None:
                            try:
                                counter_result: CounterResult = car_counter.process(frame)
                                frame_for_display = counter_result.frame
                                counted_cars = counter_result.counted_cars
                                detected_cars = counter_result.detections
                                line_y = counter_result.line_y
                            except CounterUnavailableError as exc:
                                car_counter_enabled = False
                                counted_cars = 0
                                detected_cars = 0
                                line_y = None
                                car_counter_status = f"indisponible ({exc})"
                                if controls_ui is not None:
                                    controls_ui.set_car_counter_enabled(False)
                                    counter_ui_prev = False
                                print(f"Car counter unavailable: {exc}")
                            except Exception as exc:
                                car_counter_enabled = False
                                counted_cars = 0
                                detected_cars = 0
                                line_y = None
                                car_counter_status = f"erreur runtime ({exc})"
                                if controls_ui is not None:
                                    controls_ui.set_car_counter_enabled(False)
                                    counter_ui_prev = False
                                print(f"Car counter runtime error: {exc}")

                        if not args.no_display:
                            display_frame = build_display_frame(
                                frame_for_display,
                                last_snapshot,
                                ui_state.settings,
                                ui_state.active_codec,
                                latency_avg_ms,
                                latency_jitter_ms,
                                drop_frames,
                                drop_rate_percent,
                                car_counter_enabled,
                                counted_cars,
                                detected_cars,
                                line_y,
                                car_counter_status,
                            )
                            cv2.imshow(WINDOW_NAME, display_frame)
                            key = cv2.waitKey(1) & 0xFF
                            if key == ord("q"):
                                stop_requested = True
                                break
                        received_frames += 1
                    if stop_requested:
                        break
                    if args.max_frames and received_frames >= args.max_frames:
                        break

    if controls_ui is not None:
        controls_ui.close()
    if not args.no_display:
        cv2.destroyAllWindows()

    summary = tracker.finalize()
    print(
        f"RX summary | {summary.bitrate_mbps:.2f} Mb/s average | "
        f"{summary.fps:.1f} fps average | {summary.total_megabytes:.2f} MB received | "
        f"lat={latency_avg_ms:.1f} ms | jit={latency_jitter_ms:.1f} ms | "
        f"drop={drop_rate_percent:.2f}%"
    )


if __name__ == "__main__":
    main()
