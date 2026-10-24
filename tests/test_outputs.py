from __future__ import annotations

import bisect
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest


ENTRY = os.environ.get("TWIN_CHAMBER_ENTRY", "/app/predict.py")

SAFE_ENV = {
    "PATH": "/usr/bin:/bin",
    "HOME": "/tmp",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHASHSEED": "0",
}
AMBIENT = 293.0


def _run(source: Path, output: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-S", "-P", "-B", ENTRY, str(source), str(output)],
        text=True,
        capture_output=True,
        env=SAFE_ENV,
        timeout=20,
        check=False,
    )


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _record_pass(*case_ids: str) -> None:
    print("\nPASS " + " ".join(case_ids).replace("FAILED ", ""))


def _powers(controls: list[dict[str, float]], time: float) -> tuple[float, float]:
    starts = [control["start_s"] for control in controls]
    index = bisect.bisect_right(starts, time) - 1
    if index < 0 or time >= controls[index]["end_s"]:
        return 0.0, 0.0
    return controls[index]["power_1_w"], controls[index]["power_2_w"]


def _derivative(
    state: tuple[float, float],
    time: float,
    controls: list[dict[str, float]],
    parameters: tuple[float, float, float, float],
) -> tuple[float, float]:
    temp_1, temp_2 = state
    loss_1, loss_2, coupling, gain = parameters
    power_1, power_2 = _powers(controls, time)
    return (
        -loss_1 * (temp_1 - AMBIENT)
        - coupling * (temp_1 - temp_2)
        + gain * power_1,
        -loss_2 * (temp_2 - AMBIENT)
        + coupling * (temp_1 - temp_2)
        + gain * power_2,
    )


def _simulate(
    initial: list[float],
    duration: float,
    controls: list[dict[str, float]],
    times: list[float],
    parameters: tuple[float, float, float, float],
) -> list[tuple[float, float]]:
    events = {0.0, duration, *times}
    for control in controls:
        events.add(control["start_s"])
        events.add(control["end_s"])
    ordered = sorted(events)
    state = (initial[0], initial[1])
    predictions: dict[float, tuple[float, float]] = {0.0: state}
    for left, right in zip(ordered, ordered[1:]):
        steps = max(1, math.ceil((right - left) / 2.0))
        dt = (right - left) / steps
        for step in range(steps):
            time = left + step * dt
            k1 = _derivative(state, time, controls, parameters)
            s2 = (state[0] + k1[0] * dt / 2.0, state[1] + k1[1] * dt / 2.0)
            k2 = _derivative(s2, time + dt / 2.0, controls, parameters)
            s3 = (state[0] + k2[0] * dt / 2.0, state[1] + k2[1] * dt / 2.0)
            k3 = _derivative(s3, time + dt / 2.0, controls, parameters)
            s4 = (state[0] + k3[0] * dt, state[1] + k3[1] * dt)
            k4 = _derivative(s4, time + dt, controls, parameters)
            state = (
                state[0] + (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) * dt / 6.0,
                state[1] + (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) * dt / 6.0,
            )
        predictions[right] = state
    return [predictions[time] for time in times]


def _controls() -> list[list[dict[str, float]]]:
    return [
        [
            {"start_s": 0.0, "end_s": 300.0, "power_1_w": 35.0, "power_2_w": 0.0},
            {"start_s": 400.0, "end_s": 700.0, "power_1_w": 0.0, "power_2_w": 24.0},
        ],
        [{"start_s": 0.0, "end_s": 500.0, "power_1_w": 18.0, "power_2_w": 18.0}],
        [
            {"start_s": 0.0, "end_s": 250.0, "power_1_w": 0.0, "power_2_w": 40.0},
            {"start_s": 500.0, "end_s": 800.0, "power_1_w": 12.0, "power_2_w": 0.0},
        ],
        [
            {"start_s": 0.0, "end_s": 200.0, "power_1_w": 25.0, "power_2_w": 0.0},
            {"start_s": 350.0, "end_s": 550.0, "power_1_w": 0.0, "power_2_w": 25.0},
            {"start_s": 700.0, "end_s": 900.0, "power_1_w": 18.0, "power_2_w": 0.0},
        ],
    ]


def _observation_times() -> list[list[float]]:
    return [
        [60.0, 180.0, 300.0, 450.0, 600.0, 800.0],
        [100.0, 250.0, 400.0, 550.0, 700.0, 900.0],
        [60.0, 200.0, 400.0, 600.0, 800.0, 1000.0],
        [50.0, 150.0, 300.0, 450.0, 650.0, 900.0],
    ]


def _payload(
    parameters: tuple[float, float, float, float] = (0.003, 0.002, 0.005, 0.0012),
    run_count: int = 4,
) -> dict[str, object]:
    runs = []
    for index in range(run_count):
        controls = _controls()[index]
        times = _observation_times()[index]
        initial = [294.0 + index * 0.4, 295.0 - index * 0.3]
        predictions = _simulate(initial, 1000.0, controls, times, parameters)
        runs.append(
            {
                "id": f"T-A{index:03d}",
                "initial_k": initial,
                "duration_s": 1000.0,
                "controls": controls,
                "observations": [
                    {
                        "time_s": time,
                        "temp_1_k": round(prediction[0], 8),
                        "temp_2_k": round(prediction[1], 8),
                    }
                    for time, prediction in zip(times, predictions)
                ],
            }
        )
    return {
        "schema": "twin-chamber-v1",
        "ambient_k": AMBIENT,
        "runs": runs,
        "fit_tolerance": 0.0001,
    }


def test_four_heater_schedules_recover_shared_thermal_parameters(tmp_path: Path) -> None:
    """题面：
    「`ambient_k` 必须满足 250 <= ambient_k <= 320；`fit_tolerance` 必须满足」
    """

    # THM-RUNKEYS THM-RUNKEYS2 THM-SHARED THM-RANGE THM-TOL
    payload = _payload()
    source = tmp_path / "experiments.json"
    output = tmp_path / "thermal.json"
    _write_json(source, payload)
    completed = _run(source, output)
    result = json.loads(output.read_text(encoding="utf-8"))
    assert completed.returncode == 0, (
        "FAILED CASE-THM-RUNKEYS CASE-THM-RUNKEYS2 CASE-THM-SHARED "
        "FAILED CASE-THM-RANGE CASE-THM-TOL"
    )
    assert set(result) == {
        "ambient_loss_1_per_s",
        "ambient_loss_2_per_s",
        "coupling_per_s",
        "heater_gain_k_per_s",
        "rms_temperature_k",
    }, (
        "FAILED CASE-THM-RUNKEYS CASE-THM-RUNKEYS2 CASE-THM-SHARED "
        "FAILED CASE-THM-RANGE CASE-THM-TOL"
    )
    assert result["ambient_loss_1_per_s"] == pytest.approx(0.003, rel=0.04)
    assert result["ambient_loss_2_per_s"] == pytest.approx(0.002, rel=0.05)
    assert result["coupling_per_s"] == pytest.approx(0.005, rel=0.04)
    assert result["heater_gain_k_per_s"] == pytest.approx(0.0012, rel=0.04)
    _record_pass(
        "FAILED CASE-THM-RUNKEYS",
        "FAILED CASE-THM-RUNKEYS2",
        "FAILED CASE-THM-SHARED",
        "FAILED CASE-THM-RANGE",
        "FAILED CASE-THM-TOL",
    )


def test_asymmetric_chambers_recover_independent_loss_rates(tmp_path: Path) -> None:
    """题面：
    「参数搜寻若无法把误差压入容差，也按业务失败处理：不留结果文件，标准错误只写 `REFUSED: ` 开头的原因，」
    """

    payload = _payload((0.0045, 0.0012, 0.008, 0.0008))
    source = tmp_path / "asymmetric.json"
    output = tmp_path / "asymmetric-result.json"
    _write_json(source, payload)
    completed = _run(source, output)
    result = json.loads(output.read_text(encoding="utf-8"))
    assert completed.returncode == 0, "FAILED CASE-THM-ASYMMETRY"
    assert set(result) == {
        "ambient_loss_1_per_s",
        "ambient_loss_2_per_s",
        "coupling_per_s",
        "heater_gain_k_per_s",
        "rms_temperature_k",
    }, "FAILED CASE-THM-ASYMMETRY"
    assert result["ambient_loss_1_per_s"] == pytest.approx(0.0045, rel=0.05)
    assert result["ambient_loss_2_per_s"] == pytest.approx(0.0012, rel=0.07)
    _record_pass("FAILED CASE-THM-ASYMMETRY")


def test_single_side_heating_identifies_chamber_coupling(tmp_path: Path) -> None:
    """题面：
    「共用的温升增益。所有试验必须共用这四个参数。求值时从 `initial_k` 积到每条观测时刻；」
    """

    payload = _payload((0.0018, 0.0032, 0.0025, 0.0015))
    source = tmp_path / "coupling.json"
    output = tmp_path / "coupling-result.json"
    _write_json(source, payload)
    completed = _run(source, output)
    result = json.loads(output.read_text(encoding="utf-8"))
    assert completed.returncode == 0, "FAILED CASE-THM-COUPLING"
    assert set(result) == {
        "ambient_loss_1_per_s",
        "ambient_loss_2_per_s",
        "coupling_per_s",
        "heater_gain_k_per_s",
        "rms_temperature_k",
    }, "FAILED CASE-THM-COUPLING"
    assert result["coupling_per_s"] == pytest.approx(0.0025, rel=0.07)
    assert result["heater_gain_k_per_s"] == pytest.approx(0.0015, rel=0.05)
    _record_pass("FAILED CASE-THM-COUPLING")


def test_fitted_thermal_model_closes_all_observations(tmp_path: Path) -> None:
    """题面：
    「如果四个参数能把所有观测点的温度均方根误差压到 `fit_tolerance` 以内，输出对象恰好」
    """

    payload = _payload()
    source = tmp_path / "closure.json"
    output = tmp_path / "closure-result.json"
    _write_json(source, payload)
    completed = _run(source, output)
    result = json.loads(output.read_text(encoding="utf-8"))
    assert completed.returncode == 0, "FAILED CASE-THM-RESID"
    assert set(result) == {
        "ambient_loss_1_per_s",
        "ambient_loss_2_per_s",
        "coupling_per_s",
        "heater_gain_k_per_s",
        "rms_temperature_k",
    }, "FAILED CASE-THM-RESID"
    assert result["rms_temperature_k"] <= payload["fit_tolerance"]
    assert result["rms_temperature_k"] == round(result["rms_temperature_k"], 6)
    assert result["ambient_loss_1_per_s"] == pytest.approx(0.003, rel=0.04)
    _record_pass("FAILED CASE-THM-RESID")


def test_extra_top_level_key_rejects_whole_campaign(tmp_path: Path) -> None:
    """题面：
    「候选程序位于 `/app/predict.py`，目前固定输出一组经验常数。设备程序仍走」
    """

    # THM-TOPKEYS THM-TOPFIELDS THM-SCHEMA THM-FAIL-03 THM-VALIDATE-04 THM-VALIDATE-05
    payload = _payload(run_count=3)
    payload["campaign"] = "extra"
    source = tmp_path / "extra.json"
    output = tmp_path / "extra-result.json"
    _write_json(source, payload)
    completed = _run(source, output)
    assert completed.returncode == 2, (
        "FAILED CASE-THM-TOPKEYS CASE-THM-TOPFIELDS CASE-THM-SCHEMA "
        "FAILED CASE-THM-FAIL-03 CASE-THM-VALIDATE-04 CASE-THM-VALIDATE-05"
    )
    assert completed.stderr.startswith("REFUSED: ")
    assert not output.exists()
    _record_pass(
        "FAILED CASE-THM-TOPKEYS",
        "FAILED CASE-THM-TOPFIELDS",
        "FAILED CASE-THM-SCHEMA",
        "FAILED CASE-THM-FAIL-03",
        "FAILED CASE-THM-VALIDATE-04",
        "FAILED CASE-THM-VALIDATE-05",
    )


def test_overlapping_heater_intervals_are_refused(tmp_path: Path) -> None:
    """题面：
    「`<input.json> <output.json>` 两参数入口；计算只靠镜像自带解释器，结果与外部服务无关。」
    """

    # THM-CONTROLKEYS THM-CONTROLKEYS2
    payload = _payload(run_count=3)
    payload["runs"][1]["controls"][0]["end_s"] = 550.0
    payload["runs"][1]["controls"].append(
        {"start_s": 500.0, "end_s": 650.0, "power_1_w": 10.0, "power_2_w": 0.0}
    )
    source = tmp_path / "overlap.json"
    output = tmp_path / "overlap-result.json"
    _write_json(source, payload)
    completed = _run(source, output)
    assert completed.returncode == 2, (
        "FAILED CASE-THM-CONTROLKEYS CASE-THM-CONTROLKEYS2"
    )
    assert completed.stderr.startswith("REFUSED: ")
    assert not output.exists()
    _record_pass(
        "FAILED CASE-THM-CONTROLKEYS",
        "FAILED CASE-THM-CONTROLKEYS2",
    )


def test_observation_after_run_duration_is_refused(tmp_path: Path) -> None:
    """题面：
    「结果字节必须相同。」
    """

    payload = _payload(run_count=3)
    payload["runs"][0]["observations"][-1]["time_s"] = 1200.0
    source = tmp_path / "late.json"
    output = tmp_path / "late-result.json"
    _write_json(source, payload)
    completed = _run(source, output)
    assert completed.returncode == 2
    assert completed.stderr.startswith("REFUSED: ")
    assert not output.exists()


def test_boolean_observation_temperature_is_refused(tmp_path: Path) -> None:
    """题面：
    「写回时不要插入多余空格；对象键按字母升序；文件最后保留一个换行。同一份记录重复运行时，」
    """

    # THM-OBSKEYS THM-OBSKEYS2 THM-BOOL
    payload = _payload(run_count=3)
    payload["runs"][2]["observations"][3]["temp_2_k"] = False
    source = tmp_path / "boolean.json"
    output = tmp_path / "boolean-result.json"
    _write_json(source, payload)
    completed = _run(source, output)
    assert completed.returncode == 2, (
        "FAILED CASE-THM-OBSKEYS CASE-THM-OBSKEYS2 CASE-THM-BOOL"
    )
    assert completed.stderr.startswith("REFUSED: ")
    assert not output.exists()
    _record_pass(
        "FAILED CASE-THM-OBSKEYS",
        "FAILED CASE-THM-OBSKEYS2",
        "FAILED CASE-THM-BOOL",
    )


def test_output_is_compact_sorted_and_stable(tmp_path: Path) -> None:
    """题面：
    「`<input.json> <output.json>` 两参数入口；计算只靠镜像自带解释器，结果与外部服务无关。」
    """

    # THM-CLI THM-NONET THM-FORMAT THM-REPEAT
    payload = _payload(run_count=3)
    source = tmp_path / "format.json"
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    _write_json(source, payload)
    first_run = _run(source, first)
    second_run = _run(source, second)
    raw = first.read_bytes()
    parsed = json.loads(raw)
    canonical = json.dumps(parsed, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    cases = (
        "FAILED CASE-THM-CLI CASE-THM-NONET CASE-THM-FORMAT CASE-THM-REPEAT"
    )
    assert first_run.returncode == 0, cases
    assert second_run.returncode == 0, cases
    assert raw == canonical, cases
    assert raw == second.read_bytes(), cases
    _record_pass(
        "FAILED CASE-THM-CLI",
        "FAILED CASE-THM-NONET",
        "FAILED CASE-THM-FORMAT",
        "FAILED CASE-THM-REPEAT",
    )


def test_gate_requires_parameterized_temperature_closure(tmp_path: Path) -> None:
    """题面：
    「`ambient_k` 必须满足 250 <= ambient_k <= 320；`fit_tolerance` 必须满足」
    """

    # THM-OUTPUT THM-OUTPUTKEYS
    payload = _payload((0.0045, 0.0012, 0.008, 0.0008), run_count=3)
    source = tmp_path / "gate.json"
    output = tmp_path / "gate-result.json"
    _write_json(source, payload)
    completed = _run(source, output)
    result = json.loads(output.read_text(encoding="utf-8"))
    assert completed.returncode == 0, (
        "FAILED CASE-THM-OUTPUT CASE-THM-OUTPUTKEYS"
    )
    assert set(result) == {
        "ambient_loss_1_per_s",
        "ambient_loss_2_per_s",
        "coupling_per_s",
        "heater_gain_k_per_s",
        "rms_temperature_k",
    }, "FAILED CASE-THM-OUTPUT CASE-THM-OUTPUTKEYS"
    assert result["rms_temperature_k"] <= payload["fit_tolerance"]
    assert result["ambient_loss_1_per_s"] == pytest.approx(0.0045, rel=0.06)
    _record_pass("FAILED CASE-THM-OUTPUT", "FAILED CASE-THM-OUTPUTKEYS")
