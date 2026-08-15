"""Body server: serves the viewer page and streams robot state over WebSocket.

Ownership boundary: the Python character process owns ALL state (joint
positions, light, HUD). The browser viewer is a pure display of that state —
it sends nothing back. This keeps the body swappable: the same JSON protocol
could drive a physics sim or real firmware instead of three.js.

Protocol (server -> viewer):
  {"type": "robot", ...}                    once on connect: URDF-derived description
  {"type": "state", "joints": {...},        ~30 Hz: authoritative pose + light
   "light": {"color": [r,g,b], "intensity": f}}
  {"type": "hud", "state": str,             on change: behavior state + captions
   "caption": str, "speaker": str}
"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from aiohttp import web, WSMsgType

log = logging.getLogger("body")

ROOT = Path(__file__).resolve().parent.parent


class BodyServer:
    def __init__(self, robot_description: dict, host: str = "127.0.0.1", port: int = 8765):
        self.robot_description = robot_description
        self.host, self.port = host, port
        self.clients: set[web.WebSocketResponse] = set()
        self._last_hud: dict | None = None

    async def start(self) -> None:
        app = web.Application()
        app.router.add_get("/ws", self._ws_handler)
        app.router.add_get("/", self._index)
        app.router.add_static("/assets/", ROOT / "robot" / "assets")
        app.router.add_static("/", ROOT / "viewer")
        runner = web.AppRunner(app, access_log=None)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
        log.info("viewer at http://%s:%d", self.host, self.port)

    async def _index(self, _req: web.Request) -> web.FileResponse:
        return web.FileResponse(ROOT / "viewer" / "index.html")

    async def _ws_handler(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.send_json({"type": "robot", **self.robot_description})
        if self._last_hud:
            await ws.send_json(self._last_hud)
        self.clients.add(ws)
        log.info("viewer connected (%d total)", len(self.clients))
        try:
            async for msg in ws:
                if msg.type in (WSMsgType.ERROR, WSMsgType.CLOSE):
                    break
        finally:
            self.clients.discard(ws)
        return ws

    def broadcast(self, message: dict) -> None:
        """Fire-and-forget send to all viewers; drops on slow/dead clients."""
        if message.get("type") == "hud":
            self._last_hud = message
        data = json.dumps(message)
        for ws in list(self.clients):
            if ws.closed:
                self.clients.discard(ws)
                continue
            asyncio.ensure_future(self._safe_send(ws, data))

    async def _safe_send(self, ws: web.WebSocketResponse, data: str) -> None:
        try:
            await ws.send_str(data)
        except (ConnectionResetError, RuntimeError):
            self.clients.discard(ws)
