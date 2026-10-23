#!/usr/bin/env python3
"""Starter entry point for twin-chamber identification."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 3:
        print("REFUSED: expected input and output paths", file=sys.stderr)
        return 2
    source = Path(sys.argv[1])
    target = Path(sys.argv[2])
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        print("REFUSED: unreadable input", file=sys.stderr)
        return 2
    if not isinstance(payload, dict):
        print("REFUSED: input must be an object", file=sys.stderr)
        return 2
    result = {
        "ambient_loss_1_per_s": 0.002,
        "ambient_loss_2_per_s": 0.002,
        "coupling_per_s": 0.004,
        "heater_gain_k_per_s": 0.0004,
        "rms_temperature_k": 0.0,
    }
    target.write_text(
        json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
