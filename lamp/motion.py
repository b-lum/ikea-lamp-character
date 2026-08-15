"""Animator: turns intent into joint motion the physical robot could execute.

Three layers compose every frame, then get clamped to the URDF's position and
velocity limits (single enforcement point for physical plausibility):

  1. performance layer — keyframe timelines (wake, nod, excited bounce, ...)
  2. breathing layer  — small sinusoid so the character never freezes
  3. gaze layer       — smoothed offsets aiming the head at the person

The animator owns the light too: motion and light are one expressive channel.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

from .urdf import Robot

VEL_SAFETY = 0.85  # use at most 85% of URDF velocity limit

# ---------------------------------------------------------------- easing
def ease_in_out(t: float) -> float:
    return t * t * (3 - 2 * t)

def ease_out_back(t: float) -> float:  # overshoot, reads as "springy"
    c = 2.6
    t -= 1
    return 1 + t * t * ((c + 1) * t + c)

EASES = {"smooth": ease_in_out, "spring": ease_out_back, "linear": lambda t: t}

# ---------------------------------------------------------------- poses
J = ("base_yaw_joint", "shoulder_pitch_joint", "elbow_pitch_joint",
     "neck_yaw_joint", "head_pitch_joint")

NEUTRAL = {"base_yaw_joint": 0.0, "shoulder_pitch_joint": 0.42,
           "elbow_pitch_joint": -0.85, "neck_yaw_joint": 0.0,
           "head_pitch_joint": -0.25}

DROOP = {"shoulder_pitch_joint": 0.95, "elbow_pitch_joint": -1.75,
         "neck_yaw_joint": 0.0, "head_pitch_joint": 0.65, "base_yaw_joint": 0.0}

# A performance is a list of (partial-pose, duration s, ease name).
PERFORMANCES: dict[str, list[tuple[dict, float, str]]] = {
    "wake_up": [
        (DROOP, 0.01, "linear"),
        ({"shoulder_pitch_joint": 0.75, "elbow_pitch_joint": -1.4}, 0.7, "smooth"),
        ({**NEUTRAL, "head_pitch_joint": -0.45}, 0.9, "spring"),
        (NEUTRAL, 0.5, "smooth"),
    ],
    "greet": [
        ({"head_pitch_joint": 0.15, "elbow_pitch_joint": -1.0}, 0.35, "smooth"),
        ({"head_pitch_joint": -0.45, "elbow_pitch_joint": -0.7}, 0.45, "spring"),
        (NEUTRAL, 0.5, "smooth"),
    ],
    "nod_yes": [
        ({"head_pitch_joint": 0.25}, 0.30, "smooth"),
        ({"head_pitch_joint": -0.40}, 0.32, "smooth"),
        ({"head_pitch_joint": 0.10}, 0.30, "smooth"),
        (NEUTRAL, 0.4, "smooth"),
    ],
    "shake_no": [
        ({"neck_yaw_joint": 0.5}, 0.28, "smooth"),
        ({"neck_yaw_joint": -0.5}, 0.42, "smooth"),
        ({"neck_yaw_joint": 0.25}, 0.34, "smooth"),
        (NEUTRAL, 0.4, "smooth"),
    ],
    "curious_tilt": [
        ({"neck_yaw_joint": 0.55, "head_pitch_joint": -0.5,
          "elbow_pitch_joint": -0.95}, 0.6, "spring"),
        ({}, 0.8, "linear"),  # hold
        (NEUTRAL, 0.6, "smooth"),
    ],
    "excited_bounce": [
        # anticipation crouch -> extend through the hop -> land squash -> again
        ({"shoulder_pitch_joint": 0.72, "elbow_pitch_joint": -1.35,
          "head_pitch_joint": 0.2}, 0.25, "smooth"),
        ({"shoulder_pitch_joint": 0.12, "elbow_pitch_joint": -0.4,
          "head_pitch_joint": -0.55}, 0.28, "spring"),
        ({"shoulder_pitch_joint": 0.62, "elbow_pitch_joint": -1.15,
          "head_pitch_joint": 0.1}, 0.24, "smooth"),
        ({"shoulder_pitch_joint": 0.18, "elbow_pitch_joint": -0.5,
          "head_pitch_joint": -0.5}, 0.28, "spring"),
        (NEUTRAL, 0.5, "smooth"),
    ],
    "thinking": [
        ({"neck_yaw_joint": -0.4, "head_pitch_joint": -0.55}, 0.8, "smooth"),
        ({"neck_yaw_joint": -0.25}, 1.2, "smooth"),
        ({"neck_yaw_joint": -0.45}, 1.2, "smooth"),
    ],
    "scan_scene": [
        ({"base_yaw_joint": 0.7, "head_pitch_joint": 0.05}, 1.1, "smooth"),
        ({"base_yaw_joint": -0.7}, 2.0, "smooth"),
        ({"base_yaw_joint": 0.0, **NEUTRAL}, 1.1, "smooth"),
    ],
    "sleep": [
        ({"head_pitch_joint": 0.3, "elbow_pitch_joint": -1.2}, 1.2, "smooth"),
        (DROOP, 1.8, "smooth"),
    ],
}

# Whole-body hops: display-layer character license (the URDF has no vertical
# DOF — a real unit is desk-mounted, so deployment would drop this channel).
# Each entry: (start offset s, duration s, apex height m); z follows a
# ballistic parabola so even the theatrics respect projectile physics.
HOPS: dict[str, list[tuple[float, float, float]]] = {
    "excited_bounce": [(0.25, 0.55, 0.11), (0.77, 0.50, 0.075)],
    "greet": [(0.35, 0.46, 0.06)],
}

@dataclass
class LightState:
    color: tuple[float, float, float] = (1.0, 0.91, 0.69)
    intensity: float = 0.55
    pulse_hz: float = 0.0
    pulse_depth: float = 0.0

@dataclass
class _Segment:
    start_pose: dict
    target: dict
    duration: float
    ease: str
    t0: float = 0.0


class Animator:
    """Owns authoritative joint + light state; ticked at ~30 Hz by main loop."""

    def __init__(self, robot: Robot):
        self.limits = {n: j for n, j in robot.movable_joints.items()}
        self.pos = dict(DROOP)  # boot asleep; wake_up brings it alive
        self.light = LightState(intensity=0.05)
        self._timeline: list[_Segment] = []
        self._base_pose = dict(DROOP)  # pose the performance layer resolves to
        self._gaze_target = (0.0, 0.0)  # yaw, pitch offsets (rad)
        self._gaze = [0.0, 0.0]
        self.gaze_enabled = False
        self.breathing = 1.0  # amplitude scale; 0 disables
        self._hops: list[tuple[float, float, float]] = []  # absolute t0, dur, height
        self._root_z = 0.0
        self._t = time.monotonic()

    # ---------------- intent API (called by behavior engine) ----------------
    def play(self, name: str) -> float:
        """Start a performance; returns its duration in seconds."""
        frames = PERFORMANCES[name]
        now = time.monotonic()
        self._timeline = []
        self._hops = [(now + t0, dur, h) for t0, dur, h in HOPS.get(name, [])]
        pose = dict(self._base_pose)
        t = now
        for target, dur, ease in frames:
            seg_target = {**pose, **target} if target else dict(pose)
            self._timeline.append(_Segment(dict(pose), seg_target, dur, ease, t))
            pose = seg_target
            t += dur
        return t - now

    def is_performing(self) -> bool:
        return bool(self._timeline)

    def set_gaze(self, yaw: float, pitch: float) -> None:
        self._gaze_target = (float(yaw), float(pitch))

    def set_light(self, color=None, intensity=None, pulse_hz=None, pulse_depth=None):
        if color is not None: self.light.color = tuple(color)
        if intensity is not None: self.light.intensity = intensity
        if pulse_hz is not None: self.light.pulse_hz = pulse_hz
        if pulse_depth is not None: self.light.pulse_depth = pulse_depth

    def set_pose(self, pose: dict, duration: float = 1.0) -> float:
        """Move to an explicit pose (used by goal-directed actions)."""
        now = time.monotonic()
        target = {**self._base_pose, **pose}
        self._timeline = [_Segment(dict(self._base_pose), target, duration, "smooth", now)]
        return duration

    # ---------------- tick ----------------
    def tick(self) -> dict:
        now = time.monotonic()
        dt = min(now - self._t, 0.1)
        self._t = now

        # 1. performance layer
        while self._timeline:
            seg = self._timeline[0]
            u = (now - seg.t0) / seg.duration if seg.duration > 0 else 1.0
            if u >= 1.0:
                self._base_pose = dict(seg.target)
                self._timeline.pop(0)
                continue
            e = EASES[seg.ease](u)
            self._base_pose = {
                j: seg.start_pose[j] + (seg.target[j] - seg.start_pose[j]) * e
                for j in seg.target
            }
            break

        desired = dict(self._base_pose)

        # 2. breathing layer (skips joints mid-performance for crisp keyframes)
        if self.breathing > 0 and not self._timeline:
            s = math.sin(now * 2 * math.pi * 0.22) * self.breathing
            desired["shoulder_pitch_joint"] += 0.018 * s
            desired["elbow_pitch_joint"] += 0.026 * s
            desired["head_pitch_joint"] += 0.012 * math.sin(now * 2 * math.pi * 0.22 + 0.9)

        # 3. gaze layer
        k = 1 - math.exp(-dt * 6.0)
        gy = self._gaze_target[0] if self.gaze_enabled else 0.0
        gp = self._gaze_target[1] if self.gaze_enabled else 0.0
        self._gaze[0] += (gy - self._gaze[0]) * k
        self._gaze[1] += (gp - self._gaze[1]) * k
        desired["neck_yaw_joint"] += self._gaze[0] * 0.65
        desired["base_yaw_joint"] += self._gaze[0] * 0.35
        desired["head_pitch_joint"] += self._gaze[1]

        # clamp to URDF position + velocity limits — the physical-honesty gate
        for name, joint in self.limits.items():
            target = min(max(desired[name], joint.lower), joint.upper)
            max_step = joint.velocity * VEL_SAFETY * dt
            step = min(max(target - self.pos[name], -max_step), max_step)
            self.pos[name] += step

        # hop layer (root z, ballistic parabola per hop)
        z = 0.0
        for t0, dur, h in self._hops:
            u = (now - t0) / dur
            if 0.0 <= u <= 1.0:
                z = h * 4 * u * (1 - u)
        self._hops = [hp for hp in self._hops if now < hp[0] + hp[1]]
        self._root_z = z

        # light pulse
        intensity = self.light.intensity
        if self.light.pulse_hz > 0:
            wave = 0.5 + 0.5 * math.sin(now * 2 * math.pi * self.light.pulse_hz)
            intensity *= 1.0 - self.light.pulse_depth * wave
        return {
            "type": "state",
            "joints": {n: round(p, 4) for n, p in self.pos.items()},
            "root": {"z": round(self._root_z, 4)},
            "light": {"color": [round(c, 3) for c in self.light.color],
                      "intensity": round(intensity, 3)},
        }
