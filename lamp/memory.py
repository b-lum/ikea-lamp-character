"""Scene memory: what the character has noticed about the world.

Deliberately simple — a JSON-persisted list of observations. Claude writes to
it (via the `remember` field of its act tool) and reads from it (a compact
summary is included in every prompt). Because entries carry timestamps and
positions, the character can answer "where was my mug?" even after the mug
leaves the frame.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

STORE = Path(__file__).resolve().parent.parent / "memory_store.json"
MAX_ITEMS = 40


class SceneMemory:
    def __init__(self):
        self.items: list[dict] = []
        if STORE.exists():
            try:
                self.items = json.loads(STORE.read_text())
            except json.JSONDecodeError:
                self.items = []

    def remember(self, name: str, description: str, position: str) -> None:
        self.items = [i for i in self.items if i["name"].lower() != name.lower()]
        self.items.append({"name": name, "description": description,
                           "position": position, "t": time.time()})
        self.items = self.items[-MAX_ITEMS:]
        STORE.write_text(json.dumps(self.items, indent=1))

    def summary(self) -> str:
        if not self.items:
            return "(nothing observed yet)"
        lines = []
        for i in self.items:
            age = time.time() - i["t"]
            ago = f"{age:.0f}s ago" if age < 90 else f"{age / 60:.0f}m ago"
            lines.append(f"- {i['name']}: {i['description']} (seen {i['position']}, {ago})")
        return "\n".join(lines)
