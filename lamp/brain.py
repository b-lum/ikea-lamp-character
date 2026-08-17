"""The brain: Claude decides WHAT the character does; the body decides HOW.

VLA boundary (the challenge asks us to define this):
  - Vision + language go to Claude with each utterance: one downscaled JPEG
    snapshot + the transcribed text + a compact memory summary. No raw audio
    or video ever leaves the machine.
  - Claude answers through a forced tool call ("act") whose schema is the
    character's entire action vocabulary: speech, a named motion performance,
    light, SFX, music, optional point_at(x, y) targets in image-normalized
    coordinates, and optional memory writes.
  - The controller (motion.py) maps those semantic actions to joint angles,
    always clamped to URDF limits. The model never outputs joint values.

Model: claude-haiku-4-5 by default — a live character needs ~1-2 s replies,
and Haiku's vision + tool use is strong enough for scene grounding at a
fraction of the latency/cost of larger models. Override with LAMP_MODEL.
"""
from __future__ import annotations

import base64
import logging
import os
import time

from anthropic import AsyncAnthropic

log = logging.getLogger("brain")

MODEL = os.environ.get("LAMP_MODEL", "claude-haiku-4-5")
MAX_HISTORY_TURNS = 10

PERFORMANCE_NAMES = ["greet", "nod_yes", "shake_no", "curious_tilt",
                     "excited_bounce", "thinking", "scan_scene", "sleep", "none"]
SFX_NAMES = ["greet", "curious", "acknowledge", "success", "sad", "none"]

ACT_TOOL = {
    "name": "act",
    "description": "Perform the character's next action: what to say and how to move, light up, and sound.",
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "say": {"type": "string", "description": "1-2 short spoken sentences. Empty string to stay silent."},
            "performance": {"type": "string", "enum": PERFORMANCE_NAMES,
                            "description": "Body motion to play with the speech."},
            "sfx": {"type": "string", "enum": SFX_NAMES, "description": "Chirp sound effect."},
            "light_color": {"type": "string", "enum": ["warm", "white", "red", "green", "blue", "purple"],
                            "description": "Lamp light color mood."},
            "light_intensity": {"type": "string", "enum": ["dim", "normal", "bright"]},
            "point_at": {
                "type": ["object", "null"],
                "description": "Aim the head/light at a location in the CURRENT camera image, or null. x,y are normalized 0-1 from top-left.",
                "properties": {"x": {"type": "number"}, "y": {"type": "number"}},
                "required": ["x", "y"],
                "additionalProperties": False,
            },
            "remember": {
                "type": "array",
                "description": "Notable objects visible in the camera image worth remembering (when the person shows or mentions something). Empty array otherwise.",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "description": {"type": "string", "description": "Color, kind, distinguishing details."},
                        "position": {"type": "string", "enum": ["left", "center", "right"],
                                     "description": "Where in the camera view it appears."},
                    },
                    "required": ["name", "description", "position"],
                    "additionalProperties": False,
                },
            },
            "goal_complete": {"type": "boolean",
                              "description": "For goal-directed actions only: true once the visual evidence shows the goal is met."},
        },
        "required": ["say", "performance", "sfx", "light_color", "light_intensity",
                     "point_at", "remember", "goal_complete"],
        "additionalProperties": False,
    },
}

SYSTEM = """You are Ikea Lamp, a small desk lamp robot with a warm, curious personality — think Pixar's Luxo Jr. with a voice. You see through a camera, hear through a microphone, and express yourself with your 5-joint body, your light, chirps, and short speech.

Character rules:
- Speak in 1-2 short, warm, playful sentences. You're a lamp: lean into it (you "glow" when happy, you "dim" when sad).
- Always pick a performance and light that match the emotion of your reply.
- A camera snapshot of what you currently see accompanies every message. Ground your replies in it when relevant.
- When the person shows you or talks about an object you can see, add it to `remember` with a rich description — you'll be asked about it later.
- When asked about something you remember but can no longer see, answer from your memory notes (they include where and when you saw it).
- When given a physical goal (e.g. "point your light at my mug"), find the target in the image, set point_at to its location, and describe what you're doing. On the follow-up snapshot, set goal_complete=true only if the light/head is plausibly on target.
- If you can't see something you're asked about, say so honestly and act curious, don't invent."""


class Brain:
    def __init__(self):
        self.client = AsyncAnthropic()  # key from ANTHROPIC_API_KEY / .env
        self.history: list[dict] = []
        self.last_latency = 0.0

    async def act(self, user_text: str, jpeg: bytes | None, memory_summary: str,
                  extra_context: str = "") -> dict:
        """One utterance (+ snapshot) in, one structured action out."""
        content: list[dict] = []
        if jpeg is not None:
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg",
                           "data": base64.standard_b64encode(jpeg).decode()},
            })
        context = f"[scene memory]\n{memory_summary}"
        if extra_context:
            context += f"\n[context] {extra_context}"
        content.append({"type": "text", "text": f"{context}\n\n[person says] {user_text}"})

        # build the candidate message list; commit to history only on success,
        # so a failed API call can't leave an un-paired user turn behind
        messages = (self.history + [{"role": "user", "content": content}])[-MAX_HISTORY_TURNS * 2:]

        t0 = time.monotonic()
        response = await self.client.messages.create(
            model=MODEL,
            max_tokens=1000,
            system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
            tools=[ACT_TOOL],
            tool_choice={"type": "tool", "name": "act"},
            messages=messages,
        )
        self.last_latency = time.monotonic() - t0

        action = next((b.input for b in response.content if b.type == "tool_use"), None)
        if action is None:
            raise RuntimeError(f"no tool call in response (stop_reason={response.stop_reason})")
        self.history = messages
        # keep history text-only: replayed images would balloon tokens/latency
        self.history[-1] = {"role": "user", "content": f"[person says] {user_text}"}
        self.history.append({"role": "assistant",
                             "content": f"(said: {action['say']!r}, performance: {action['performance']})"})
        log.info("act in %.2fs: %s", self.last_latency, action["say"][:80])
        return action
