"""Entry point.

  python -m lamp.main                # full character
  python -m lamp.main --demo-motion  # body only: cycle performances in the viewer
"""
from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

from . import urdf
from .body_server import BodyServer
from .motion import Animator, PERFORMANCES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)-10s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("main")

ROOT = Path(__file__).resolve().parent.parent
TICK_HZ = 30


async def animation_loop(animator: Animator, server: BodyServer) -> None:
    while True:
        server.broadcast(animator.tick())
        await asyncio.sleep(1 / TICK_HZ)


async def demo_motion(animator: Animator, server: BodyServer) -> None:
    """Cycle every performance so the body can be verified without the brain."""
    await asyncio.sleep(2)  # give the viewer a moment to connect
    animator.set_light(intensity=0.7)
    while True:
        for name in PERFORMANCES:
            log.info("performance: %s", name)
            server.broadcast({"type": "hud", "state": "demo", "caption": name, "speaker": "performance"})
            duration = animator.play(name)
            await asyncio.sleep(duration + 1.0)


async def run(args: argparse.Namespace) -> None:
    robot = urdf.load(ROOT / "robot" / "dummy_lamp_5dof.urdf")
    log.info("robot '%s': %d movable joints", robot.name, len(robot.movable_joints))
    animator = Animator(robot)
    server = BodyServer(robot.to_viewer_dict())
    await server.start()
    tasks = [asyncio.create_task(animation_loop(animator, server))]

    if args.demo_motion:
        tasks.append(asyncio.create_task(demo_motion(animator, server)))
    else:
        from .behavior import Character
        character = Character(animator, server)
        tasks.append(asyncio.create_task(character.run()))

    await asyncio.gather(*tasks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Lamp character")
    parser.add_argument("--demo-motion", action="store_true",
                        help="cycle motion performances without perception/AI")
    args = parser.parse_args()
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        log.info("bye")


if __name__ == "__main__":
    main()
