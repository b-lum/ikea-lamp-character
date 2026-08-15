# Technical note — Lux, a live lamp character

## Architecture and data flow

```
                        ┌─────────────── character process (Python, asyncio) ───────────────┐
 laptop camera ──► perception thread ── engagement events ──►┐                              │
                   (YuNet face det.,     gaze target         │                              │
                    ~10 fps, local)                          ▼                              │
 laptop mic ────► hearing thread ─── utterance text ──► BEHAVIOR ENGINE ──► animator ───────┼──► WebSocket ──► browser viewer
                   (energy VAD +                        state machine       30 Hz tick      │    JSON @30Hz    (three.js body,
                    Whisper base.en,                    asleep/engaged/     URDF pos+vel    │                  built from URDF)
                    local)                              thinking/acting     limits clamp    │
                                                          │      ▲                          │
 snapshot JPEG + text + memory summary ───────────────────┘      │                          │
                   ▼                                             │                          │
            Anthropic API (claude-haiku-4-5) ── act tool ────────┘                          │
            speech/performance/light/sfx/point_at/remember       ▼                          │
                                                     audio out (TTS, synthesized           │
 scene memory (JSON, persistent) ◄──────────────►    SFX + music, auto-ducking) ──► speaker │
                        └───────────────────────────────────────────────────────────────────┘
```

One Python process owns all state; the browser is a pure display. The URDF is
parsed once and is the single source of truth: the viewer builds its scene
from the parsed description sent on connect, and the animator enforces the
same file's joint position *and velocity* limits on every 30 Hz tick.

## Protocol

JSON over one WebSocket. Downstream (authoritative state): `robot` (once,
URDF-derived description), `state` (30 Hz: joint positions, root hop offset,
light color/intensity), `hud` (behavior state + captions), `camera` (5 Hz
annotated webcam PiP). Upstream, exactly one event: `viewpoint` — the orbit
camera's azimuth, which the character treats as "where the audience window
is" and turns toward (through real joints, still limit-clamped). The same
protocol could drive a physics sim or firmware; the viewer is swappable.

## Model-to-action (the VLA boundary)

Every utterance goes to Claude with: transcribed text, one downscaled JPEG
snapshot, and a compact scene-memory summary. Claude must reply through a
forced, strictly-validated `act` tool whose schema is the character's entire
action vocabulary: `say`, a named motion performance, light color/intensity,
SFX, optional `point_at(x, y)` in **image-normalized coordinates**, memory
writes, and `goal_complete`. The model decides *what* (semantic actions
grounded in pixels); the controller decides *how* (maps image coordinates to
joint angles, clamped to URDF limits). The model never outputs joint values.
Goal-directed actions loop: act → move → fresh snapshot → Claude verifies its
own aim before setting `goal_complete`.

## Key choices and tradeoffs

- **Kinematic animation, no physics engine.** The robot is a desk-mounted arm
  whose dynamics are dominated by its servos; a physics sim adds cost and
  failure modes without adding believability. Physical plausibility is instead
  enforced where it matters: per-tick clamping to the URDF's position and
  velocity limits (85% derating), spring easing, and layered idle motion.
  Tradeoff: no contact/collision simulation.
- **Custom three.js viewer over PyBullet/MuJoCo.** Chosen for expressive
  rendering (real spotlight, emissive bulb, theming) and because the WebSocket
  boundary makes the architecture explicit. The lamp's whole-body hop is a
  deliberate character license on a separate display-only channel (`root.z`,
  ballistic profile); a real desk-mounted unit would drop that channel. The
  viewer also hides the URDF's camera-marker nub and restyles materials —
  the robot model itself is unmodified.
- **Hybrid local/cloud AI.** Engagement, VAD, and STT run locally (YuNet
  230 KB; faster-whisper base.en int8) — the character stays responsive and
  private; only utterance text + one JPEG per exchange reach the cloud.
  Claude (`claude-haiku-4-5`) handles vision grounding, dialogue, memory
  curation, and goal planning in a single call per utterance — chosen over
  larger models because conversational latency (~1–2 s) matters more to a
  live character than reasoning depth. Model is a config knob.
- **All SFX and music synthesized in numpy at startup** — no assets, no
  licensing, and the music mixer auto-ducks under speech.
- **Self-echo prevention:** the mic is gated while TTS plays, so the
  character never transcribes its own voice.

## Deployment (Ubuntu 24.04 target)

Plain Python venv + `apt` packages (README has the exact commands); no GPU,
no containers needed at this scale. CPU-heavy components were sized for 4
cores / 8 GB: YuNet at ~10 fps, Whisper base.en int8 (loads in <2 s after
first download), 30 Hz animation. TTS backend auto-selects: `piper` (local
neural TTS) on Linux, `say` on macOS dev machines. The viewer vendors all JS
dependencies, so the system runs fully offline except Anthropic API calls.

## Measurements (MacBook dev machine, single process)

| Metric | Value |
|---|---|
| Engagement: camera→wake reaction | < 1 s (0.7 s debounce + detector at ~10 fps) |
| STT latency (utterance end → text) | ~0.5–1.5 s (base.en int8, CPU) |
| LLM latency (snapshot+text → action) | ~1–2 s (claude-haiku-4-5, forced tool) |
| End-to-end utterance → response start | ~2–4 s |
| CPU (whole process, engaged + tracking) | ~40–80% of one core |
| Memory (RSS) | ~280–420 MB |
| Animation tick | 30 Hz; viewer renders at display rate |

Latencies are recorded live to `metrics.json` by the built-in instrumentation.

## Known limitations / intentionally left out

- Aiming uses a fixed camera→joint mapping (laptop camera ≈ lamp's forward
  view), not calibrated hand-eye geometry; goal verification by re-observation
  compensates for coarse aim.
- Engagement is face-presence-based; YuNet's frontal bias approximates "looking
  toward" but there is no true gaze estimation.
- Single-person interaction; no face identification across sessions.
- No physics/collision sim (see tradeoff above); no IK solver — expressive
  poses are authored, and pointing uses a 2-DOF mapping.
- Energy-based VAD is tuned for quiet rooms; noisy environments would need
  a neural VAD (silero) — left out for scope.
