"""Runtime measurements for the technical note: latencies + CPU/RSS."""
from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from pathlib import Path

import psutil

log = logging.getLogger("metrics")
OUT = Path(__file__).resolve().parent.parent / "metrics.json"


class Metrics:
    def __init__(self):
        self.samples: dict[str, list[float]] = defaultdict(list)
        self._proc = psutil.Process()
        self._proc.cpu_percent()  # prime the counter

    def record(self, name: str, seconds: float) -> None:
        self.samples[name].append(round(seconds, 3))

    def sample_system(self) -> None:
        self.samples["cpu_percent"].append(self._proc.cpu_percent())
        self.samples["rss_mb"].append(round(self._proc.memory_info().rss / 1e6, 1))

    def summary(self) -> dict:
        out = {}
        for name, vals in self.samples.items():
            if vals:
                s = sorted(vals)
                out[name] = {"n": len(vals), "mean": round(sum(vals) / len(vals), 3),
                             "p50": s[len(s) // 2], "max": max(vals)}
        return out

    def dump(self) -> None:
        OUT.write_text(json.dumps({"summary": self.summary(), "samples": self.samples}, indent=1))
        log.info("metrics: %s", self.summary())
