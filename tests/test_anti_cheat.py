from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).parent))
from test_outputs import ENTRY, _payload, _run, _write_json  # noqa: E402


def _solve(
    tmp_path: Path,
    label: str,
    parameters: tuple[float, float, float, float],
    run_count: int,
) -> dict[str, float]:
    payload = _payload(parameters, run_count=run_count)
    source = tmp_path / f"{label}.json"
    output = tmp_path / f"{label}-result.json"
    _write_json(source, payload)
    completed = _run(source, output)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == ""
    assert completed.stderr == ""
    return json.loads(output.read_text(encoding="utf-8"))


def test_two_campaigns_require_real_parameter_identification(tmp_path: Path) -> None:
    first = _solve(tmp_path, "campaign-a", (0.0042, 0.0014, 0.0065, 0.0009), 3)
    second = _solve(tmp_path, "campaign-b", (0.0019, 0.0031, 0.0022, 0.0016), 4)

    assert first["ambient_loss_1_per_s"] == pytest.approx(0.0042, rel=0.08)
    assert first["ambient_loss_2_per_s"] == pytest.approx(0.0014, rel=0.08)
    assert first["coupling_per_s"] == pytest.approx(0.0065, rel=0.08)
    assert first["heater_gain_k_per_s"] == pytest.approx(0.0009, rel=0.08)

    assert second["ambient_loss_1_per_s"] == pytest.approx(0.0019, rel=0.08)
    assert second["ambient_loss_2_per_s"] == pytest.approx(0.0031, rel=0.08)
    assert second["coupling_per_s"] == pytest.approx(0.0022, rel=0.08)
    assert second["heater_gain_k_per_s"] == pytest.approx(0.0016, rel=0.08)

    assert first != second

