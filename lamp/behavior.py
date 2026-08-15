"""Behavior engine: the one place where perception, hearing, brain, body, and
sound come together into a single character.

State machine:

  ASLEEP --(face engaged)--> ENGAGED --(utterance)--> THINKING --> ACTING --+
     ^                          ^                                           |
     |                          +-------------------------------------------+
     +--(face lost a while)-----+

While ENGAGED the ears are open, ambient music plays softly, and the gaze
layer keeps the head on the person. Every utterance goes to the brain with a
live snapshot; the returned action drives voice, motion, light, and SFX
together so responses read as one intentional gesture.
"""
from __future__ import annotations

import asyncio
import logging
import time

from .audio_in import Hearing
from .audio_out import AudioOut
from .brain import Brain
from .memory import SceneMemory
from .metrics import Metrics
from .motion import NEUTRAL, Animator
from .perception import Perception

log = logging.getLogger("character")

LIGHT_COLORS = {
    "warm": (1.0, 0.83, 0.55), "white": (1.0, 0.98, 0.92), "red": (1.0, 0.35, 0.30),
    "green": (0.45, 1.0, 0.55), "blue": (0.45, 0.65, 1.0), "purple": (0.75, 0.50, 1.0),
}
LIGHT_LEVELS = {"dim": 0.25, "normal": 0.6, "bright": 1.0}

# camera-image position -> aim offsets (radians); laptop camera ~60 deg hfov
POINT_YAW_RANGE = 1.0
POINT_PITCH_RANGE = 0.6
GOAL_MAX_FOLLOWUPS = 2


class Character:
    def __init__(self, animator: Animator, server):
        self.animator = animator
        self.server = server
        self.memory = SceneMemory()
        self.brain = Brain()
        self.metrics = Metrics()
        self.audio = AudioOut()
        self.state = "asleep"
        self.events: asyncio.Queue = asyncio.Queue()
        self._loop = asyncio.get_event_loop()
        self._pointing_until = 0.0
        self._busy = False

        self.perception = Perception(on_event=self._thread_event)
        self.hearing = Hearing(on_utterance=self._thread_utterance,
                               is_self_speaking=self.audio._speaking.is_set)

    # ---------- thread -> asyncio bridges ----------
    def _thread_event(self, name: str) -> None:
        self._loop.call_soon_threadsafe(self.events.put_nowait, ("face", name))

    def _thread_utterance(self, text: str, seconds: float) -> None:
        self.metrics.record("stt_latency", self.hearing.last_stt_latency)
        self._loop.call_soon_threadsafe(self.events.put_nowait, ("utterance", text))

    # ---------- helpers ----------
    def _hud(self, state: str | None = None, caption: str = "", speaker: str = "") -> None:
        msg = {"type": "hud", "caption": caption, "speaker": speaker}
        if state:
            msg["state"] = state
        self.server.broadcast(msg)

    async def _speak(self, text: str) -> None:
        if not text:
            return
        self._hud(caption=text, speaker="lux")
        t0 = time.monotonic()
        await asyncio.get_event_loop().run_in_executor(None, self.audio.speak, text)
        self.metrics.record("tts_duration", time.monotonic() - t0)

    def _apply_light(self, action: dict) -> None:
        self.animator.set_light(color=LIGHT_COLORS[action["light_color"]],
                                intensity=LIGHT_LEVELS[action["light_intensity"]],
                                pulse_hz=0, pulse_depth=0)

    def _point_at(self, x: float, y: float) -> float:
        """Aim head + light at an image-normalized location. Returns move time."""
        yaw = (0.5 - x) * POINT_YAW_RANGE
        pitch = (y - 0.5) * POINT_PITCH_RANGE
        pose = {**NEUTRAL,
                "base_yaw_joint": yaw * 0.6,
                "neck_yaw_joint": yaw * 0.4,
                "head_pitch_joint": NEUTRAL["head_pitch_joint"] + pitch}
        duration = self.animator.set_pose(pose, 1.4)
        self._pointing_until = time.monotonic() + duration + 6.0
        self.animator.gaze_enabled = False
        return duration

    async def _execute(self, action: dict) -> None:
        """Turn one brain action into synchronized motion+light+sfx+speech."""
        for item in action.get("remember") or []:
            self.memory.remember(item["name"], item["description"], item["position"])
            log.info("remembered: %s (%s)", item["name"], item["position"])
        self._apply_light(action)
        if action.get("sfx") not in (None, "none"):
            self.audio.play_sfx(action["sfx"])
        move_time = 0.0
        if action.get("point_at"):
            move_time = self._point_at(action["point_at"]["x"], action["point_at"]["y"])
        elif action.get("performance") not in (None, "none"):
            self.animator.play(action["performance"])
        speak_task = asyncio.create_task(self._speak(action["say"]))
        if move_time:
            await asyncio.sleep(move_time)
        await speak_task

    # ---------- state transitions ----------
    async def _engage(self) -> None:
        self.state = "engaged"
        self._hud(state="engaged")
        self.audio.play_sfx("wake")
        duration = self.animator.play("wake_up")
        self.animator.set_light(color=LIGHT_COLORS["warm"], intensity=0.6,
                                pulse_hz=0, pulse_depth=0)
        await asyncio.sleep(duration * 0.7)
        self.animator.gaze_enabled = True
        self.audio.set_music(True, 0.55)
        self.animator.play("greet")
        await self._speak("Oh, hello! I'm Lux.")
        self.hearing.enabled = True
        self._hud(state="listening")

    async def _disengage(self) -> None:
        self.state = "asleep"
        self.hearing.enabled = False
        self.animator.gaze_enabled = False
        self.audio.set_music(False)
        self.audio.play_sfx("sleep")
        self._hud(state="idle", caption="")
        self.animator.set_light(intensity=0.05)
        self.animator.play("sleep")

    async def _handle_utterance(self, text: str) -> None:
        self._busy = True
        try:
            self._hud(state="thinking", caption=text, speaker="you")
            self.animator.play("thinking")
            self.animator.set_light(pulse_hz=1.2, pulse_depth=0.5)
            t0 = time.monotonic()
            action = await self.brain.act(text, self.perception.snapshot_jpeg(),
                                          self.memory.summary())
            self.metrics.record("llm_latency", self.brain.last_latency)
            self._hud(state="responding")
            await self._execute(action)
            self.metrics.record("utterance_to_response", time.monotonic() - t0)

            # goal-directed loop: after acting on the scene, re-observe & verify
            followups = 0
            while (action.get("point_at") and not action.get("goal_complete")
                   and followups < GOAL_MAX_FOLLOWUPS):
                followups += 1
                await asyncio.sleep(1.0)  # let the world (and our head) settle
                self._hud(state="verifying")
                action = await self.brain.act(
                    "(no speech — this is your follow-up look at the scene)",
                    self.perception.snapshot_jpeg(), self.memory.summary(),
                    extra_context="You just moved. Check the new snapshot: is your "
                                  "head/light aimed at the target? If yes set "
                                  "goal_complete=true and celebrate briefly; if not, "
                                  "adjust point_at.")
                self.metrics.record("llm_latency", self.brain.last_latency)
                await self._execute(action)
            if action.get("goal_complete"):
                self.audio.play_sfx("success")
            self._hud(state="listening")
        except Exception:
            log.exception("utterance handling failed")
            self.audio.play_sfx("sad")
            await self._speak("Oh no, my thoughts flickered. Say that again?")
            self._hud(state="listening")
        finally:
            self._busy = False

    # ---------- tasks ----------
    async def _gaze_task(self) -> None:
        while True:
            if self.state == "engaged" and time.monotonic() > self._pointing_until:
                self.animator.gaze_enabled = True
                s = self.perception.state
                if s.engaged:
                    self.animator.set_gaze(-0.55 * s.face_x, 0.35 * s.face_y)
            await asyncio.sleep(0.1)

    async def _camera_pip_task(self) -> None:
        """Stream a small annotated webcam view so 'what Lux sees' is visible."""
        import base64
        while True:
            if self.server.clients:
                jpeg = self.perception.annotated_jpeg()
                if jpeg:
                    self.server.broadcast({"type": "camera",
                                           "data": base64.standard_b64encode(jpeg).decode()})
            await asyncio.sleep(0.2)

    async def _metrics_task(self) -> None:
        while True:
            await asyncio.sleep(10)
            self.metrics.sample_system()
            self.metrics.record("camera_fps", self.perception.state.fps)
            self.metrics.dump()

    async def run(self) -> None:
        self.perception.start()
        self.hearing.start()
        self._hud(state="idle")
        asyncio.create_task(self._gaze_task())
        asyncio.create_task(self._camera_pip_task())
        asyncio.create_task(self._metrics_task())
        log.info("character alive — waiting for a face")

        while True:
            kind, payload = await self.events.get()
            if kind == "face" and payload == "engaged" and self.state == "asleep":
                await self._engage()
            elif kind == "face" and payload == "disengaged" and self.state == "engaged":
                if not self._busy:
                    await self._disengage()
            elif kind == "utterance" and self.state == "engaged":
                await self._handle_utterance(payload)
