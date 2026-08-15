"""Audio output: voice (TTS), sound effects, and music.

Every sound effect and the music loop are synthesized with numpy at startup —
no audio assets, no licensing, a few ms of CPU. One sounddevice output stream
mixes music + SFX; the voice goes through the platform TTS engine and the
music auto-ducks while the character speaks.

TTS backends (selected at runtime):
  darwin -> `say` (built into macOS, zero setup for the dev/demo machine)
  linux  -> `piper` CLI (local neural TTS, CPU-friendly on the Ubuntu target)
"""
from __future__ import annotations

import logging
import platform
import shutil
import subprocess
import threading

import numpy as np
import sounddevice as sd

log = logging.getLogger("audio")

SR = 24000

# ------------------------------------------------------------------ synthesis
def _env(n: int, attack: float = 0.01, release: float = 0.08) -> np.ndarray:
    e = np.ones(n)
    a = min(int(attack * SR), n // 2)
    r = min(int(release * SR), n - a)
    if a: e[:a] = np.linspace(0, 1, a)
    if r: e[-r:] *= np.linspace(1, 0, r)
    return e

def _tone(freq: float, dur: float, vol: float = 0.5, shape: str = "sine") -> np.ndarray:
    t = np.arange(int(dur * SR)) / SR
    if shape == "sine":
        w = np.sin(2 * np.pi * freq * t)
    else:  # soft triangle-ish
        w = np.sin(2 * np.pi * freq * t) + 0.3 * np.sin(4 * np.pi * freq * t)
    return (w * _env(len(t)) * vol).astype(np.float32)

def _sweep(f0: float, f1: float, dur: float, vol: float = 0.5) -> np.ndarray:
    t = np.arange(int(dur * SR)) / SR
    phase = 2 * np.pi * (f0 * t + (f1 - f0) * t * t / (2 * dur))
    return (np.sin(phase) * _env(len(t)) * vol).astype(np.float32)

def _seq(*parts: np.ndarray) -> np.ndarray:
    return np.concatenate(parts)

def _pluck(freq: float, dur: float, vol: float = 0.35) -> np.ndarray:
    t = np.arange(int(dur * SR)) / SR
    w = np.sin(2 * np.pi * freq * t) * np.exp(-t * 6)
    w += 0.4 * np.sin(2 * np.pi * freq * 2 * t) * np.exp(-t * 9)
    return (w * vol).astype(np.float32)

def build_sfx() -> dict[str, np.ndarray]:
    """R2D2-adjacent chirps; each maps to a character intent."""
    return {
        "wake": _seq(_sweep(300, 900, 0.18), _sweep(700, 1400, 0.22, 0.4)),
        "greet": _seq(_tone(880, 0.09), _tone(1175, 0.09), _tone(1568, 0.16, 0.4)),
        "curious": _seq(_sweep(600, 1100, 0.22, 0.35), _sweep(1100, 850, 0.16, 0.3)),
        "acknowledge": _seq(_tone(1047, 0.07, 0.35), _tone(1319, 0.11, 0.35)),
        "thinking": _seq(_tone(523, 0.1, 0.2), _tone(659, 0.1, 0.2), _tone(587, 0.14, 0.2)),
        "success": _seq(_pluck(523, 0.2), _pluck(659, 0.2), _pluck(784, 0.2), _pluck(1047, 0.5, 0.45)),
        "sad": _seq(_sweep(700, 480, 0.3, 0.3), _sweep(480, 320, 0.4, 0.25)),
        "sleep": _seq(_sweep(800, 500, 0.35, 0.25), _sweep(500, 250, 0.6, 0.2)),
    }

def build_music_loop() -> np.ndarray:
    """Warm ambient loop: pad chords + pentatonic plucks, 8 bars at 88 BPM."""
    rng = np.random.default_rng(7)
    beat = 60 / 88
    bar = beat * 4
    total = int(8 * bar * SR)
    mix = np.zeros(total, dtype=np.float32)

    chords = [(220.0, 261.63, 329.63), (174.61, 220.0, 261.63),
              (196.0, 246.94, 293.66), (146.83, 220.0, 293.66)]
    for b in range(8):
        chord = chords[b % 4]
        start = int(b * bar * SR)
        n = int(bar * SR)
        t = np.arange(n) / SR
        pad = sum(np.sin(2 * np.pi * f * t) * 0.05 for f in chord)
        pad += sum(np.sin(2 * np.pi * f * 0.5 * t) * 0.03 for f in chord[:1])
        fade = np.minimum(1, np.minimum(t / 0.4, (bar - t) / 0.6))
        mix[start:start + n] += (pad * fade).astype(np.float32)

    penta = [440.0, 523.25, 587.33, 659.26, 783.99, 880.0]
    for b in range(8):
        for step in range(4):
            if rng.random() < 0.55:
                note = penta[rng.integers(len(penta))]
                p = _pluck(note, beat * 1.5, 0.16)
                start = int((b * bar + step * beat) * SR)
                end = min(start + len(p), total)
                mix[start:end] += p[:end - start]
    return np.clip(mix, -1, 1)


# ------------------------------------------------------------------ mixer
class AudioOut:
    def __init__(self):
        self.sfx = build_sfx()
        self.music = build_music_loop()
        self._music_pos = 0
        self.music_on = False
        self._music_gain = 0.0        # smoothed toward _music_target
        self._music_target = 0.0
        self._active: list[list] = []  # [array, position]
        self._lock = threading.Lock()
        self._speaking = threading.Event()
        self._stream = sd.OutputStream(samplerate=SR, channels=1, dtype="float32",
                                       blocksize=1024, callback=self._callback)
        self._stream.start()
        self._tts_backend = self._pick_tts()
        log.info("audio out ready (tts=%s)", self._tts_backend)

    def _pick_tts(self) -> str:
        if platform.system() == "Darwin" and shutil.which("say"):
            return "say"
        if shutil.which("piper"):
            return "piper"
        return "none"

    def _callback(self, out: np.ndarray, frames: int, _time, _status) -> None:
        buf = np.zeros(frames, dtype=np.float32)
        target = self._music_target * (0.25 if self._speaking.is_set() else 1.0)
        self._music_gain += (target - self._music_gain) * 0.05
        if self._music_gain > 1e-3:
            idx = (self._music_pos + np.arange(frames)) % len(self.music)
            buf += self.music[idx] * self._music_gain
        self._music_pos = (self._music_pos + frames) % len(self.music)
        with self._lock:
            for item in self._active:
                arr, pos = item
                chunk = arr[pos:pos + frames]
                buf[:len(chunk)] += chunk
                item[1] += frames
            self._active = [i for i in self._active if i[1] < len(i[0])]
        out[:, 0] = np.clip(buf, -1, 1)

    # ---------------- intent API ----------------
    def play_sfx(self, name: str) -> None:
        with self._lock:
            self._active.append([self.sfx[name], 0])

    def set_music(self, on: bool, gain: float = 0.9) -> None:
        self.music_on = on
        self._music_target = gain if on else 0.0

    def speak(self, text: str) -> None:
        """Blocking TTS (run via executor). Music ducks automatically."""
        self._speaking.set()
        try:
            if self._tts_backend == "say":
                subprocess.run(["say", "-v", "Samantha", "-r", "178", text], check=False)
            elif self._tts_backend == "piper":
                subprocess.run(
                    "piper --model /usr/local/share/piper/en_US-amy-medium.onnx "
                    "--output-raw | aplay -r 22050 -f S16_LE -t raw -",
                    input=text.encode(), shell=True, check=False)
            else:
                log.warning("no TTS backend; wanted to say: %s", text)
        finally:
            self._speaking.clear()
