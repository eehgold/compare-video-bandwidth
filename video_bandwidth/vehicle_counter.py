from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - optional dependency at runtime
    YOLO = None


class CounterUnavailableError(RuntimeError):
    """Raised when the car counter backend is unavailable."""


@dataclass(slots=True)
class _Track:
    track_id: int
    cx: float
    cy: float
    last_seen_frame: int
    counted: bool = False


@dataclass(slots=True)
class CounterResult:
    frame: np.ndarray
    line_y: int
    counted_cars: int
    detections: int


class CarCounter:
    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        confidence: float = 0.35,
        iou: float = 0.45,
        line_ratio: float = 0.82,
        max_distance: float = 85.0,
        max_missed_frames: int = 12,
    ) -> None:
        self._model_name = model_name
        self._confidence = confidence
        self._iou = iou
        self._line_ratio = line_ratio
        self._max_distance = max_distance
        self._max_missed_frames = max_missed_frames
        self._model: Any | None = None
        self._next_track_id = 1
        self._frame_index = 0
        self._counted_cars = 0
        self._tracks: dict[int, _Track] = {}

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        if YOLO is None:
            raise CounterUnavailableError(
                "ultralytics is not installed. Install dependencies with: "
                "python -m pip install -r requirements.txt"
            )
        try:
            self._model = YOLO(self._model_name)
        except Exception as exc:  # pragma: no cover - depends on local runtime
            raise CounterUnavailableError(
                f"Unable to load YOLO model '{self._model_name}': {exc}"
            ) from exc

    def reset(self) -> None:
        self._next_track_id = 1
        self._frame_index = 0
        self._counted_cars = 0
        self._tracks.clear()

    @property
    def counted_cars(self) -> int:
        return self._counted_cars

    def _detect(self, frame: np.ndarray) -> list[tuple[float, float, float, float]]:
        self._ensure_model()
        result = self._model(
            frame,
            verbose=False,
            classes=[2],  # COCO class id for car
            conf=self._confidence,
            iou=self._iou,
        )[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.cpu().numpy()
        return [
            (float(x1), float(y1), float(x2), float(y2))
            for x1, y1, x2, y2 in xyxy
        ]

    def _match_track(self, cx: float, cy: float, used_track_ids: set[int]) -> int | None:
        best_track_id: int | None = None
        best_distance = float("inf")
        for track_id, track in self._tracks.items():
            if track_id in used_track_ids:
                continue
            distance = ((cx - track.cx) ** 2 + (cy - track.cy) ** 2) ** 0.5
            if distance < self._max_distance and distance < best_distance:
                best_distance = distance
                best_track_id = track_id
        return best_track_id

    def process(self, frame: np.ndarray) -> CounterResult:
        self._frame_index += 1
        line_y = int(frame.shape[0] * self._line_ratio)
        detections = self._detect(frame)
        used_track_ids: set[int] = set()
        tracked_boxes: list[tuple[int, tuple[float, float, float, float]]] = []

        for box in detections:
            x1, y1, x2, y2 = box
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            track_id = self._match_track(cx, cy, used_track_ids)
            if track_id is None:
                track_id = self._next_track_id
                self._next_track_id += 1
                self._tracks[track_id] = _Track(
                    track_id=track_id,
                    cx=cx,
                    cy=cy,
                    last_seen_frame=self._frame_index,
                    counted=False,
                )
            else:
                track = self._tracks[track_id]
                previous_y = track.cy
                moving_down = cy > previous_y
                crossed_line = previous_y < line_y <= cy
                if crossed_line and moving_down and not track.counted:
                    track.counted = True
                    self._counted_cars += 1
                track.cx = cx
                track.cy = cy
                track.last_seen_frame = self._frame_index

            used_track_ids.add(track_id)
            tracked_boxes.append((track_id, box))

        stale_ids = [
            track_id
            for track_id, track in self._tracks.items()
            if self._frame_index - track.last_seen_frame > self._max_missed_frames
        ]
        for track_id in stale_ids:
            self._tracks.pop(track_id, None)

        frame_out = frame.copy()
        cv2.line(
            frame_out,
            (0, line_y),
            (frame_out.shape[1], line_y),
            (0, 0, 255),
            2,
        )
        for track_id, box in tracked_boxes:
            x1, y1, x2, y2 = [int(v) for v in box]
            cv2.rectangle(frame_out, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame_out,
                f"car #{track_id}",
                (x1, max(y1 - 8, 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        cv2.putText(
            frame_out,
            f"Cars counted: {self._counted_cars}",
            (20, max(line_y - 12, 22)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        return CounterResult(
            frame=frame_out,
            line_y=line_y,
            counted_cars=self._counted_cars,
            detections=len(tracked_boxes),
        )
