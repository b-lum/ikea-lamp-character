"""Perception: laptop camera -> engagement signal + gaze target + frames.

Everything here runs locally (YuNet face detector, ~230 KB ONNX on CPU).
Camera frames only leave the machine when the brain deliberately snapshots
the scene for Claude — engagement/attention never depends on the network.

Engagement heuristic (debounced both ways so the character doesn't flicker):
  - a sufficiently large, sufficiently confident face sustained for ENGAGE_S
    seconds => person is here and facing us (YuNet only fires on near-frontal
    faces, which doubles as a cheap "looking toward the character" test)
  - no such face for DISENGAGE_S seconds => attention moved elsewhere
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger("percept")

MODEL = Path(__file__).resolve().parent.parent / "models" / "face_detection_yunet_2023mar.onnx"
FRAME_W, FRAME_H = 640, 480
MIN_FACE_FRAC = 0.06   # face width as fraction of frame width ("close enough")
ENGAGE_S = 0.7
DISENGAGE_S = 2.5


@dataclass
class PerceptionState:
    engaged: bool = False
    face_x: float = 0.0      # face center, normalized [-1, 1], +right
    face_y: float = 0.0      # normalized [-1, 1], +down
    face_frac: float = 0.0   # face width / frame width
    fps: float = 0.0


class Perception:
    def __init__(self, camera_index: int = 0, on_event=None):
        """on_event(name: str) is called from the camera thread on
        'engaged' / 'disengaged' transitions; caller bridges to asyncio."""
        self.state = PerceptionState()
        self.on_event = on_event
        self._camera_index = camera_index
        self._latest_frame: np.ndarray | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="perception", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def latest_frame(self) -> np.ndarray | None:
        """Most recent BGR frame (for brain snapshots)."""
        with self._lock:
            return None if self._latest_frame is None else self._latest_frame.copy()

    def snapshot_jpeg(self, quality: int = 85, max_w: int = 1024) -> bytes | None:
        frame = self.latest_frame()
        if frame is None:
            return None
        h, w = frame.shape[:2]
        if w > max_w:
            frame = cv2.resize(frame, (max_w, int(h * max_w / w)))
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return buf.tobytes() if ok else None

    # ------------------------------------------------------------- thread
    def _run(self) -> None:
        cap = cv2.VideoCapture(self._camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
        if not cap.isOpened():
            log.error("camera %d failed to open", self._camera_index)
            return
        detector = cv2.FaceDetectorYN.create(str(MODEL), "", (FRAME_W, FRAME_H),
                                             score_threshold=0.7)
        log.info("camera open, YuNet loaded")

        face_since = face_lost_since = None
        t_prev = time.monotonic()
        while not self._stop.is_set():
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.1)
                continue
            frame = cv2.resize(frame, (FRAME_W, FRAME_H))
            with self._lock:
                self._latest_frame = frame

            detector.setInputSize((FRAME_W, FRAME_H))
            _, faces = detector.detect(frame)
            now = time.monotonic()
            self.state.fps = 0.9 * self.state.fps + 0.1 * (1.0 / max(now - t_prev, 1e-3))
            t_prev = now

            best = None
            if faces is not None and len(faces):
                best = max(faces, key=lambda f: f[2] * f[3])  # largest by area
            close = best is not None and best[2] / FRAME_W >= MIN_FACE_FRAC

            if close:
                x, y, w, h = (float(v) for v in best[:4])  # numpy -> plain floats
                cx, cy = (x + w / 2) / FRAME_W, (y + h / 2) / FRAME_H
                self.state.face_x = cx * 2 - 1
                self.state.face_y = cy * 2 - 1
                self.state.face_frac = w / FRAME_W
                face_lost_since = None
                face_since = face_since or now
                if not self.state.engaged and now - face_since >= ENGAGE_S:
                    self.state.engaged = True
                    log.info("engaged (face %.0f%% of frame)", self.state.face_frac * 100)
                    if self.on_event:
                        self.on_event("engaged")
            else:
                face_since = None
                self.state.face_frac = 0.0
                face_lost_since = face_lost_since or now
                if self.state.engaged and now - face_lost_since >= DISENGAGE_S:
                    self.state.engaged = False
                    log.info("disengaged")
                    if self.on_event:
                        self.on_event("disengaged")

            time.sleep(0.03)  # ~15-20 fps effective; plenty for engagement
        cap.release()
