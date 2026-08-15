"""Audio input: laptop mic -> VAD -> local Whisper transcription.

Speech-to-text runs entirely locally (faster-whisper base.en, int8, CPU) —
raw microphone audio never leaves the machine; only the transcribed text is
later sent to the language model.

The energy VAD tracks an adaptive noise floor, so it works across rooms
without tuning; captured utterances are gain-normalized before Whisper.
While the character itself is speaking, capture is gated off to stop it from
transcribing its own voice through the laptop speakers. The input device can
be switched live via set_device().
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque

import numpy as np
import sounddevice as sd

log = logging.getLogger("hearing")

SR = 16000
BLOCK = 1600            # 100 ms
START_RATIO = 2.2       # speech when rms > floor * ratio
START_MIN_S = 0.15
END_SILENCE_S = 0.8
MIN_UTTERANCE_S = 0.5
MAX_UTTERANCE_S = 15.0
PRE_ROLL_BLOCKS = 3     # keep 300 ms before trigger so first word isn't clipped


class Hearing:
    def __init__(self, on_utterance, is_self_speaking=lambda: False, device: int | None = None):
        """on_utterance(text, seconds) called from worker thread.
        is_self_speaking() gates capture during TTS playback."""
        self.on_utterance = on_utterance
        self.is_self_speaking = is_self_speaking
        self.enabled = False       # behavior engine opens/closes the ears
        self.device = device       # None = system default
        self.level = 0.0           # live mic level (0-1, floor-relative) for the UI meter
        self.speech = False        # True while an utterance is being captured
        self.last_stt_latency = 0.0
        self._blocks: deque[np.ndarray] = deque()
        self._cv = threading.Condition()
        self._stop = threading.Event()
        self._reopen = threading.Event()
        self._model = None

    def start(self) -> None:
        threading.Thread(target=self._worker, name="hearing", daemon=True).start()

    def stop(self) -> None:
        self._stop.set()
        with self._cv:
            self._cv.notify()

    def set_device(self, device: int | None) -> None:
        self.device = device
        self._reopen.set()
        with self._cv:
            self._cv.notify()

    def _callback(self, indata: np.ndarray, _frames, _time, _status) -> None:
        if self.is_self_speaking():
            return
        with self._cv:
            self._blocks.append(indata[:, 0].copy())
            self._cv.notify()

    def _worker(self) -> None:
        from faster_whisper import WhisperModel
        t0 = time.monotonic()
        self._model = WhisperModel("base.en", device="cpu", compute_type="int8")
        log.info("whisper base.en loaded in %.1fs", time.monotonic() - t0)

        while not self._stop.is_set():
            try:
                stream = sd.InputStream(samplerate=SR, channels=1, dtype="float32",
                                        blocksize=BLOCK, device=self.device,
                                        callback=self._callback)
                stream.start()
            except Exception as e:
                log.error("mic open failed (device=%s): %s — falling back to default",
                          self.device, e)
                self.device = None
                time.sleep(1)
                continue
            name = sd.query_devices(self.device if self.device is not None else None,
                                    kind="input")["name"]
            log.info("listening via mic: %s", name)
            self._reopen.clear()
            with self._cv:
                self._blocks.clear()
            self._capture_loop()
            stream.stop()
            stream.close()

    def _capture_loop(self) -> None:
        noise_floor = 0.01
        pre_roll: deque[np.ndarray] = deque(maxlen=PRE_ROLL_BLOCKS)
        utterance: list[np.ndarray] = []
        speech_run = silence_run = 0.0
        in_speech = False

        while not self._stop.is_set() and not self._reopen.is_set():
            with self._cv:
                while not self._blocks:
                    self._cv.wait(timeout=0.5)
                    if self._stop.is_set() or self._reopen.is_set():
                        return
                block = self._blocks.popleft()
            rms = float(np.sqrt(np.mean(block ** 2)))
            block_s = len(block) / SR
            self.level = max(0.0, min(1.0, (rms - noise_floor) / (noise_floor * 6 + 1e-6)))
            self.speech = in_speech

            if not in_speech:
                # only adapt the floor to non-speech audio
                noise_floor = 0.98 * noise_floor + 0.02 * max(rms, 1e-4)

            if not self.enabled:
                pre_roll.append(block)
                in_speech, utterance = False, []
                continue

            is_speech = rms > noise_floor * START_RATIO
            if not in_speech:
                pre_roll.append(block)
                speech_run = speech_run + block_s if is_speech else 0.0
                if speech_run >= START_MIN_S:
                    in_speech, silence_run = True, 0.0
                    utterance = list(pre_roll)
                    log.info("speech detected (rms %.4f, floor %.4f) — capturing", rms, noise_floor)
            else:
                utterance.append(block)
                silence_run = 0.0 if is_speech else silence_run + block_s
                total = sum(len(b) for b in utterance) / SR
                if silence_run >= END_SILENCE_S or total >= MAX_UTTERANCE_S:
                    in_speech, speech_run = False, 0.0
                    if total - silence_run >= MIN_UTTERANCE_S:
                        self._transcribe(np.concatenate(utterance))
                    utterance = []

    def _transcribe(self, audio: np.ndarray) -> None:
        t0 = time.monotonic()
        peak = float(np.abs(audio).max())
        if peak > 1e-4:  # auto-gain: laptop mics run quiet; whisper likes ~full-scale
            audio = np.clip(audio * min(0.9 / peak, 20.0), -1.0, 1.0)
        segments, _info = self._model.transcribe(audio, language="en", beam_size=1,
                                                 vad_filter=True)
        text = " ".join(s.text.strip() for s in segments).strip()
        self.last_stt_latency = time.monotonic() - t0
        if text:
            log.info("heard (%.2fs stt): %s", self.last_stt_latency, text)
            self.on_utterance(text, len(audio) / SR)
        else:
            log.info("captured %.1fs but whisper found no speech — try speaking "
                     "louder or closer to the mic", len(audio) / SR)
