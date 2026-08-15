"""Device discovery + persisted selection (camera / mic / speaker).

macOS quirk that motivated this: Continuity Camera can make an iPhone camera
index 0, hijacking the default. Cameras are probed with a snapshot thumbnail
(OpenCV exposes no device names), audio devices by name via sounddevice.
Selections persist in devices.json and apply live.
"""
from __future__ import annotations

import base64
import json
import logging
from pathlib import Path

import cv2
import sounddevice as sd

log = logging.getLogger("devices")
CFG = Path(__file__).resolve().parent.parent / "devices.json"
MAX_CAMERA_INDEX = 4


def load() -> dict:
    try:
        return json.loads(CFG.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def save(cfg: dict) -> None:
    CFG.write_text(json.dumps(cfg, indent=1))


def _thumb(frame) -> str:
    h, w = frame.shape[:2]
    frame = cv2.resize(frame, (192, int(h * 192 / w)))
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
    return base64.standard_b64encode(buf.tobytes()).decode() if ok else ""


def snapshot(perception, hearing, audio) -> dict:
    """Full device inventory + current selections. Blocking — run in executor."""
    cameras = []
    for i in range(MAX_CAMERA_INDEX + 1):
        if i == perception._camera_index:
            frame = perception.latest_frame()
            cameras.append({"index": i, "current": True,
                            "thumb": _thumb(frame) if frame is not None else ""})
            continue
        cap = cv2.VideoCapture(i)
        ok, frame = cap.read() if cap.isOpened() else (False, None)
        cap.release()
        if ok:
            cameras.append({"index": i, "current": False, "thumb": _thumb(frame)})

    audio_in, audio_out = [], []
    default_in, default_out = sd.default.device
    for d in sd.query_devices():
        entry = {"index": d["index"], "name": d["name"]}
        if d["max_input_channels"] > 0:
            audio_in.append({**entry, "current": d["index"] == (hearing.device
                             if hearing.device is not None else default_in)})
        if d["max_output_channels"] > 0:
            audio_out.append({**entry, "current": d["index"] == (audio.device
                              if audio.device is not None else default_out)})
    return {"cameras": cameras, "audio_in": audio_in, "audio_out": audio_out}
