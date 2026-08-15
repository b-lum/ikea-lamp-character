# Lux — a live lamp robot character

A 5-DOF desk-lamp robot character built for the Live Character Robot Software
Challenge. Lux sees through the laptop camera, listens through the microphone,
speaks through the speaker, and performs with motion, light, chirps, and music
in a browser-rendered 3D body driven over WebSocket.

![architecture](docs/architecture.png)

See [docs/TECHNICAL_NOTE.md](docs/TECHNICAL_NOTE.md) for architecture,
design decisions, and measurements.

## What it does

- **Engagement** — wakes with a chirp and greeting when a person looks at it
  (local YuNet face detection, debounced), tracks their face with its head,
  and goes back to sleep when they leave.
- **Spoken interaction** — local Whisper transcription in, platform TTS out;
  ambient music ducks while anyone is speaking.
- **Scene memory** — objects it sees are written to a persistent JSON store
  and recalled later ("where was my mug?").
- **Goal-directed action** — "point your light at the bottle": Claude locates
  the target in the camera image, the controller aims the head, and a fresh
  snapshot verifies the aim before Lux declares success.

## Setup — Ubuntu 24.04 (target environment)

```bash
sudo apt update && sudo apt install -y python3-venv python3-dev \
    libportaudio2 libgl1 libglib2.0-0 alsa-utils

git clone <this repo> && cd lamp-character
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# local TTS voice (the app auto-detects piper on Linux, `say` on macOS)
.venv/bin/pip install piper-tts
sudo mkdir -p /usr/local/share/piper
sudo curl -L -o /usr/local/share/piper/en_US-amy-medium.onnx \
  "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx"
sudo curl -L -o /usr/local/share/piper/en_US-amy-medium.onnx.json \
  "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx.json"

cp env.example .env   # then paste your Anthropic API key into .env
```

The first run downloads the Whisper `base.en` model (~150 MB) automatically.

## Setup — macOS (development)

Same as above minus `apt`/piper (uses the built-in `say` voice):

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp env.example .env   # add your Anthropic API key
```

## Run

```bash
.venv/bin/python -m lamp.main            # full character
.venv/bin/python -m lamp.main --demo-motion  # body only, no camera/mic/API
```

Open **http://127.0.0.1:8765** to see the body. Grant camera and microphone
permission on first run. Runtime measurements accumulate in `metrics.json`.

Configuration lives in `.env`:

| Variable | Default | Meaning |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | required for the brain |
| `LAMP_MODEL` | `claude-haiku-4-5` | vision/dialogue model (latency-optimized) |

## Repository layout

```
lamp/                 character process (Python)
  main.py             entry point, 30 Hz animation loop
  urdf.py             URDF parser — single source of truth for the robot
  body_server.py      WebSocket/HTTP body server + JSON protocol
  motion.py           layered animator: performances + breathing + gaze,
                      URDF position/velocity limits enforced every tick
  perception.py       camera -> engagement + gaze (local YuNet)
  audio_in.py         mic -> VAD -> local Whisper
  audio_out.py        TTS + synthesized SFX/music mixer
  brain.py            Claude: structured `act` tool = the VLA boundary
  memory.py           persistent scene memory
  behavior.py         state machine tying everything together
  metrics.py          latency/CPU/RSS instrumentation
viewer/               three.js body viewer (vendored deps, fully offline)
robot/                supplied URDF + mesh (unmodified)
docs/                 technical note
```

## Privacy / data flow

Face detection, voice detection, and speech-to-text all run **locally**.
The only data sent to the cloud (Anthropic API) is: transcribed utterance
text, one downscaled JPEG snapshot per utterance, and the scene-memory
summary. No audio or video streams ever leave the machine.
