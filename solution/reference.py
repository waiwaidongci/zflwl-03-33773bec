#!/usr/bin/env python3
"""Reference solver for the twin-chamber thermal inverse problem."""

from __future__ import annotations

import bisect
import json
import math
import re
import sys
from pathlib import Path
from typing import Any


TOP_KEYS = {"schema", "ambient_k", "runs", "fit_tolerance"}
RUN_KEYS = {"id", "initial_k", "duration_s", "controls", "observations"}
CONTROL_KEYS = {"start_s", "end_s", "power_1_w", "power_2_w"}
OBSERVATION_KEYS = {"time_s", "temp_1_k", "temp_2_k"}
RUN_ID = re.compile(r"T-[A-Z0-9]{4}\Z")
LOW = (1e-5, 1e-5, 1e-5, 1e-6)
HIGH = (0.05, 0.05, 0.05, 0.02)


class Refused(ValueError):
    pass


def _number(value: Any, label: str, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Refused(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number) or not low <= number <= high:
        raise Refused(f"{label} is outside its allowed range")
    return number


def parse_input(path: Path) -> tuple[float, float, list[dict[str, Any]]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise Refused("input cannot be read") from exc
    except json.JSONDecodeError as exc:
        raise Refused("input is not valid JSON") from exc
    if not isinstance(payload, dict) or set(payload) != TOP_KEYS:
        raise Refused("top-level keys do not match twin-chamber-v1")
    if payload.get("schema") != "twin-chamber-v1":
        raise Refused("unsupported schema")
    ambient = _number(payload.get("ambient_k"), "ambient_k", 250.0, 320.0)
    tolerance = _number(payload.get("fit_tolerance"), "fit_tolerance", 1e-4, 0.5)
    raw_runs = payload.get("runs")
    if not isinstance(raw_runs, list) or not 3 <= len(raw_runs) <= 8:
        raise Refused("runs must contain between 3 and 8 entries")

    runs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for run_index, raw in enumerate(raw_runs):
        if not isinstance(raw, dict) or set(raw) != RUN_KEYS:
            raise Refused(f"run {run_index} has an invalid key set")
        run_id = raw.get("id")
        if not isinstance(run_id, str) or RUN_ID.fullmatch(run_id) is None:
            raise Refused(f"run {run_index} has an invalid id")
        if run_id in seen:
            raise Refused("run ids must be unique")
        seen.add(run_id)
        raw_initial = raw.get("initial_k")
        if not isinstance(raw_initial, list) or len(raw_initial) != 2:
            raise Refused(f"{run_id} initial_k must contain two temperatures")
        initial = [
            _number(value, f"{run_id} initial temperature", 250.0, 500.0)
            for value in raw_initial
        ]
        duration = _number(raw.get("duration_s"), f"{run_id} duration_s", 120.0, 1800.0)

        raw_controls = raw.get("controls")
        if not isinstance(raw_controls, list) or not 1 <= len(raw_controls) <= 12:
            raise Refused(f"{run_id} controls has an invalid length")
        controls: list[dict[str, float]] = []
        previous_end = 0.0
        for control_index, raw_control in enumerate(raw_controls):
            if not isinstance(raw_control, dict) or set(raw_control) != CONTROL_KEYS:
                raise Refused(f"{run_id} control {control_index} has an invalid key set")
            start = _number(
                raw_control.get("start_s"), "control start", 0.0, duration
            )
            end = _number(raw_control.get("end_s"), "control end", 0.0, duration)
            if start < previous_end or end <= start:
                raise Refused(f"{run_id} controls must be ordered and non-overlapping")
            previous_end = end
            controls.append(
                {
                    "start_s": start,
                    "end_s": end,
                    "power_1_w": _number(
                        raw_control.get("power_1_w"), "power_1_w", 0.0, 60.0
                    ),
                    "power_2_w": _number(
                        raw_control.get("power_2_w"), "power_2_w", 0.0, 60.0
                    ),
                }
            )

        raw_observations = raw.get("observations")
        if not isinstance(raw_observations, list) or not 6 <= len(raw_observations) <= 40:
            raise Refused(f"{run_id} observations has an invalid length")
        observations: list[dict[str, float]] = []
        previous_time = -1.0
        for observation_index, raw_observation in enumerate(raw_observations):
            if (
                not isinstance(raw_observation, dict)
                or set(raw_observation) != OBSERVATION_KEYS
            ):
                raise Refused(
                    f"{run_id} observation {observation_index} has an invalid key set"
                )
            time = _number(
                raw_observation.get("time_s"), "observation time", 0.0, duration
            )
            if time <= previous_time:
                raise Refused(f"{run_id} observation times must strictly increase")
            previous_time = time
            observations.append(
                {
                    "time_s": time,
                    "temp_1_k": _number(
                        raw_observation.get("temp_1_k"), "temp_1_k", 250.0, 500.0
                    ),
                    "temp_2_k": _number(
                        raw_observation.get("temp_2_k"), "temp_2_k", 250.0, 500.0
                    ),
                }
            )
        runs.append(
            {
                "id": run_id,
                "initial": initial,
                "duration": duration,
                "controls": controls,
                "observations": observations,
            }
        )
    return ambient, tolerance, runs


def _powers(controls: list[dict[str, float]], time: float) -> tuple[float, float]:
    starts = [control["start_s"] for control in controls]
    index = bisect.bisect_right(starts, time) - 1
    if index < 0:
        return 0.0, 0.0
    control = controls[index]
    if time >= control["end_s"]:
        return 0.0, 0.0
    return control["power_1_w"], control["power_2_w"]


def _derivative(
    state: tuple[float, float],
    time: float,
    ambient: float,
    controls: list[dict[str, float]],
    parameters: tuple[float, float, float, float],
) -> tuple[float, float]:
    temp_1, temp_2 = state
    loss_1, loss_2, coupling, gain = parameters
    power_1, power_2 = _powers(controls, time)
    heat_1 = (
        -loss_1 * (temp_1 - ambient)
        - coupling * (temp_1 - temp_2)
        + gain * power_1
    )
    heat_2 = (
        -loss_2 * (temp_2 - ambient)
        + coupling * (temp_1 - temp_2)
        + gain * power_2
    )
    return heat_1, heat_2


def simulate(
    run: dict[str, Any],
    ambient: float,
    parameters: tuple[float, float, float, float],
) -> list[tuple[float, float]]:
    observations = run["observations"]
    events = {0.0, run["duration"]}
    events.update(control["start_s"] for control in run["controls"])
    events.update(control["end_s"] for control in run["controls"])
    events.update(observation["time_s"] for observation in observations)
    ordered = sorted(events)
    state = (run["initial"][0], run["initial"][1])
    predictions: dict[float, tuple[float, float]] = {0.0: state}
    for left, right in zip(ordered, ordered[1:]):
        steps = max(1, int(math.ceil((right - left) / 2.0)))
        dt = (right - left) / steps
        for step in range(steps):
            time = left + step * dt
            k1 = _derivative(state, time, ambient, run["controls"], parameters)
            s2 = (state[0] + k1[0] * dt / 2.0, state[1] + k1[1] * dt / 2.0)
            k2 = _derivative(s2, time + dt / 2.0, ambient, run["controls"], parameters)
            s3 = (state[0] + k2[0] * dt / 2.0, state[1] + k2[1] * dt / 2.0)
            k3 = _derivative(s3, time + dt / 2.0, ambient, run["controls"], parameters)
            s4 = (state[0] + k3[0] * dt, state[1] + k3[1] * dt)
            k4 = _derivative(s4, time + dt, ambient, run["controls"], parameters)
            state = (
                state[0] + (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) * dt / 6.0,
                state[1] + (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) * dt / 6.0,
            )
        predictions[right] = state
    return [predictions[observation["time_s"]] for observation in observations]


def _parameters(z: list[float]) -> tuple[float, float, float, float]:
    return tuple(
        LOW[index] * (HIGH[index] / LOW[index]) ** z[index]
        for index in range(4)
    )


def _residuals(
    z: list[float],
    ambient: float,
    runs: list[dict[str, Any]],
) -> list[float]:
    parameters = _parameters(z)
    residuals: list[float] = []
    for run in runs:
        predictions = simulate(run, ambient, parameters)
        for prediction, observation in zip(predictions, run["observations"]):
            residuals.append(prediction[0] - observation["temp_1_k"])
            residuals.append(prediction[1] - observation["temp_2_k"])
    return residuals


def _solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    size = len(vector)
    work = [row[:] + [value] for row, value in zip(matrix, vector)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(work[row][column]))
        if abs(work[pivot][column]) < 1e-16:
            raise Refused("thermal fit became singular")
        work[column], work[pivot] = work[pivot], work[column]
        scale = work[column][column]
        for item in range(column, size + 1):
            work[column][item] /= scale
        for row in range(size):
            if row == column:
                continue
            factor = work[row][column]
            for item in range(column, size + 1):
                work[row][item] -= factor * work[column][item]
    return [work[row][size] for row in range(size)]


def _fit_start(
    start: list[float],
    ambient: float,
    runs: list[dict[str, Any]],
) -> tuple[float, list[float]]:
    z = [min(1.0, max(0.0, value)) for value in start]
    residuals = _residuals(z, ambient, runs)
    sse = math.fsum(value * value for value in residuals)
    damping = 1e-3
    for _ in range(65):
        jacobian = [[0.0 for _ in range(4)] for _ in residuals]
        for column in range(4):
            step = 1e-5
            plus = z[:]
            minus = z[:]
            plus[column] = min(1.0, plus[column] + step)
            minus[column] = max(0.0, minus[column] - step)
            span = plus[column] - minus[column]
            if span <= 1e-12:
                continue
            r_plus = _residuals(plus, ambient, runs)
            r_minus = _residuals(minus, ambient, runs)
            for row in range(len(residuals)):
                jacobian[row][column] = (r_plus[row] - r_minus[row]) / span
        normal = [[0.0 for _ in range(4)] for _ in range(4)]
        gradient = [0.0 for _ in range(4)]
        for row, residual in enumerate(residuals):
            for left in range(4):
                gradient[left] += jacobian[row][left] * residual
                for right in range(4):
                    normal[left][right] += jacobian[row][left] * jacobian[row][right]
        for index in range(4):
            normal[index][index] += damping * max(1.0, normal[index][index])
        try:
            delta = _solve(normal, [-value for value in gradient])
        except Refused:
            break
        candidate = [min(1.0, max(0.0, z[index] + delta[index])) for index in range(4)]
        candidate_residuals = _residuals(candidate, ambient, runs)
        candidate_sse = math.fsum(value * value for value in candidate_residuals)
        movement = math.sqrt(sum((candidate[i] - z[i]) ** 2 for i in range(4)))
        if candidate_sse < sse:
            z = candidate
            residuals = candidate_residuals
            sse = candidate_sse
            damping = max(1e-9, damping / 3.0)
            if movement < 1e-9:
                break
        else:
            damping *= 8.0
            if damping > 1e9:
                break
    return sse, z


def _z_from_parameters(values: tuple[float, float, float, float]) -> list[float]:
    return [
        math.log(values[index] / LOW[index])
        / math.log(HIGH[index] / LOW[index])
        for index in range(4)
    ]


def fit(
    ambient: float,
    tolerance: float,
    runs: list[dict[str, Any]],
) -> dict[str, float]:
    starts = [
        _z_from_parameters((0.0030, 0.0020, 0.0050, 0.0004)),
        _z_from_parameters((0.0015, 0.0035, 0.0020, 0.0010)),
        _z_from_parameters((0.0060, 0.0010, 0.0100, 0.0002)),
        _z_from_parameters((0.0022, 0.0026, 0.0040, 0.0006)),
    ]
    candidates = [_fit_start(start, ambient, runs) for start in starts]
    sse, z = min(candidates, key=lambda item: item[0])
    observations = sum(len(run["observations"]) * 2 for run in runs)
    rms = math.sqrt(sse / observations)
    if rms > tolerance:
        raise Refused("thermal model cannot reach the requested fit tolerance")
    values = _parameters(z)
    return {
        "ambient_loss_1_per_s": round(values[0], 6),
        "ambient_loss_2_per_s": round(values[1], 6),
        "coupling_per_s": round(values[2], 6),
        "heater_gain_k_per_s": round(values[3], 6),
        "rms_temperature_k": round(rms, 6),
    }


def main() -> int:
    if len(sys.argv) != 3:
        print("REFUSED: expected input and output paths", file=sys.stderr)
        return 2
    source = Path(sys.argv[1])
    target = Path(sys.argv[2])
    try:
        ambient, tolerance, runs = parse_input(source)
        result = fit(ambient, tolerance, runs)
    except Refused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    target.write_text(
        json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
