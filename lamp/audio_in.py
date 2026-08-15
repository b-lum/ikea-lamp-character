"""Audio input: laptop mic -> VAD -> local Whisper transcription.

Speech-to-text runs entirely locally (faster-whisper base.en, int8, CPU) —
raw microphone audio never leaves the machine; only the transcribed text is
later sent to the language model.

The energy VAD tracks an adaptive noise floor, so it works across rooms
without tuning. While the character itself is speaking, capture is gated off
to stop it from transcribing its own voice through the laptop speakers.
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
START_RATIO = 3.5       # speech when rms > floor * ratio
START_MIN_S = 0.2
END_SILENCE_S = 0.8
MIN_UTTERANCE_S = 0.5
MAX_UTTERANCE_S = 15.0
PRE_ROLL_BLOCKS = 3     # keep 300 ms before trigger so first word isn't clipped


class Hearing:
    def __init__(self, on_utterance, is_self_speaking=lambda: False):
        """on_utterance(text, seconds) called from worker thread.
        is_self_speaking() gates capture during TTS playback."""
        self.on_utterance = on_utterance
        self.is_self_speaking = is_self_speaking
        self.enabled = False       # behavior engine opens/closes the ears
        self.last_stt_latency = 0.0
        self._blocks: deque[np.ndarray] = deque()
        self._cv = threading.Condition()
        self._stop = threading.Event()
        self._model = None

    def start(self) -> None:
        threading.Thread(target=self._worker, name="hearing", daemon=True).start()

    def stop(self) -> None:
        self._stop.set()

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

        stream = sd.InputStream(samplerate=SR, channels=1, dtype="float32",
                                blocksize=BLOCK, callback=self._callback)
        stream.start()

        noise_floor = 0.01
        pre_roll: deque[np.ndarray] = deque(maxlen=PRE_ROLL_BLOCKS)
        utterance: list[np.ndarray] = []
        speech_run = silence_run = 0.0
        in_speech = False

        while not self._stop.is_set():
            with self._cv:
                while not self._blocks:
                    self._cv.wait(timeout=0.5)
                    if self._stop.is_set():
                        return
                block = self._blocks.popleft()
            rms = float(np.sqrt(np.mean(block ** 2)))
            block_s = len(block) / SR

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
        segments, _info = self._model.transcribe(audio, language="en", beam_size=1,
                                                 vad_filter=True)
        text = " ".join(s.text.strip() for s in segments).strip()
        self.last_stt_latency = time.monotonic() - t0
        if text:
            log.info("heard (%.2fs stt): %s", self.last_stt_latency, text)
            self.on_utterance(text, len(audio) / SR)
