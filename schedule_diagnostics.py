"""Preprint-aligned schedule diagnostics and artifact persistence."""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch as th


@dataclass(frozen=True)
class RoundDiagnostics:
    """Diagnostics used to optimize one discretized annealing schedule."""

    schedule_before: np.ndarray
    schedule_after: np.ndarray
    energy_increments: np.ndarray
    length_increments: np.ndarray
    cumulative_length: np.ndarray
    normalized_cumulative_length: np.ndarray
    v_infinity: np.ndarray
    diffusion_time: np.ndarray
    guidance_weight: np.ndarray
    sigma: np.ndarray
    v_eff: np.ndarray
    v_dom: np.ndarray
    total_energy: float
    total_length: float


def build_round_diagnostics(
    *,
    schedule: np.ndarray,
    energy_increments: np.ndarray,
    v_infinity: np.ndarray,
    diffusion_time: np.ndarray,
    guidance_weight: np.ndarray,
    sigma: np.ndarray,
    v_eff: np.ndarray,
    v_dom: np.ndarray,
) -> RoundDiagnostics:
    """Compute length increments and the constant-speed schedule update."""
    schedule = np.asarray(schedule, dtype=np.float64)
    energy_increments = np.asarray(energy_increments, dtype=np.float64)
    v_infinity = np.asarray(v_infinity, dtype=np.float64)
    diffusion_time = np.asarray(diffusion_time, dtype=np.float64)
    guidance_weight = np.asarray(guidance_weight, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    v_eff = np.asarray(v_eff, dtype=np.float64)
    v_dom = np.asarray(v_dom, dtype=np.float64)

    if schedule.ndim != 1 or schedule.size < 2:
        raise ValueError("schedule must be one-dimensional with at least two points.")
    interval_count = schedule.size - 1
    for name, values in (
        ("energy_increments", energy_increments),
        ("v_infinity", v_infinity),
    ):
        if values.shape != (interval_count,):
            raise ValueError(
                f"{name} must have shape ({interval_count},), got {values.shape}."
            )
    for name, values in (
        ("diffusion_time", diffusion_time),
        ("guidance_weight", guidance_weight),
        ("sigma", sigma),
        ("v_eff", v_eff),
        ("v_dom", v_dom),
    ):
        if values.shape != schedule.shape:
            raise ValueError(
                f"{name} must have shape {schedule.shape}, got {values.shape}."
            )
    if not np.all(np.isfinite(energy_increments)):
        raise ValueError("energy_increments contains non-finite values.")
    if not np.all(np.isfinite(schedule)):
        raise ValueError("schedule contains non-finite values.")
    if np.any(np.diff(schedule) < 0.0):
        raise ValueError("schedule must be monotonically non-decreasing.")
    if np.any(energy_increments < -1e-12):
        raise ValueError("energy_increments must be non-negative.")

    energy_increments = np.maximum(energy_increments, 0.0)
    length_increments = np.sqrt(energy_increments)
    cumulative_length = np.concatenate(
        [np.zeros(1, dtype=np.float64), np.cumsum(length_increments)]
    )
    total_energy = float(energy_increments.sum())
    total_length = float(cumulative_length[-1])

    if total_length <= 0.0:
        normalized_cumulative_length = np.zeros_like(cumulative_length)
        schedule_after = schedule.copy()
    else:
        normalized_cumulative_length = cumulative_length / total_length
        unique_length, unique_indices = np.unique(
            normalized_cumulative_length,
            return_index=True,
        )
        unique_schedule = schedule[unique_indices]
        if unique_length[-1] < 1.0:
            unique_length = np.append(unique_length, 1.0)
            unique_schedule = np.append(unique_schedule, schedule[-1])

        uniform_length = np.linspace(0.0, 1.0, schedule.size)
        schedule_after = np.interp(
            uniform_length,
            unique_length,
            unique_schedule,
        )
        schedule_after[0] = schedule[0]
        schedule_after[-1] = schedule[-1]

    return RoundDiagnostics(
        schedule_before=schedule.copy(),
        schedule_after=schedule_after,
        energy_increments=energy_increments,
        length_increments=length_increments,
        cumulative_length=cumulative_length,
        normalized_cumulative_length=normalized_cumulative_length,
        v_infinity=v_infinity,
        diffusion_time=diffusion_time,
        guidance_weight=guidance_weight,
        sigma=sigma,
        v_eff=v_eff,
        v_dom=v_dom,
        total_energy=total_energy,
        total_length=total_length,
    )


def save_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Save a JSON object, converting path-like values to strings."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, default=str) + "\n")


def save_round_artifacts(
    *,
    run_directory: Path,
    round_index: int,
    diagnostics: RoundDiagnostics,
    samples: th.Tensor,
    metadata: Mapping[str, Any],
) -> Path:
    """Save all arrays and samples produced by one optimization round."""
    round_directory = run_directory / f"round_{round_index:04d}"
    round_directory.mkdir(parents=True, exist_ok=True)

    array_artifacts = {
        "schedule_before": diagnostics.schedule_before,
        "schedule_after": diagnostics.schedule_after,
        "energy_increments": diagnostics.energy_increments,
        "length_increments": diagnostics.length_increments,
        "cumulative_length": diagnostics.cumulative_length,
        "normalized_cumulative_length": diagnostics.normalized_cumulative_length,
        "v_infinity": diagnostics.v_infinity,
        "diffusion_time": diagnostics.diffusion_time,
        "guidance_weight": diagnostics.guidance_weight,
        "sigma": diagnostics.sigma,
        "v_eff": diagnostics.v_eff,
        "v_dom": diagnostics.v_dom,
    }
    for name, values in array_artifacts.items():
        np.save(round_directory / f"{name}.npy", values)

    np.savez_compressed(
        round_directory / "diagnostics.npz",
        **array_artifacts,
        total_energy=np.asarray(diagnostics.total_energy),
        total_length=np.asarray(diagnostics.total_length),
    )
    th.save(samples.detach().cpu(), round_directory / "samples.pt")
    save_json(
        round_directory / "summary.json",
        {
            **metadata,
            "round_index": round_index,
            "total_energy": diagnostics.total_energy,
            "total_length": diagnostics.total_length,
            "num_schedule_points": diagnostics.schedule_before.size,
            "num_samples": samples.shape[0],
        },
    )
    return round_directory


def save_run_history(
    *,
    run_directory: Path,
    schedule_history: Sequence[np.ndarray],
    diagnostics_history: Sequence[RoundDiagnostics],
) -> None:
    """Save aggregate histories and the latest optimized schedule."""
    run_directory.mkdir(parents=True, exist_ok=True)
    schedules = np.stack(schedule_history)
    np.save(run_directory / "schedule_history.npy", schedules)
    np.save(run_directory / "latest_schedule.npy", schedules[-1])

    if not diagnostics_history:
        return

    array_histories = {
        "energy_increment_history": np.stack(
            [item.energy_increments for item in diagnostics_history]
        ),
        "length_increment_history": np.stack(
            [item.length_increments for item in diagnostics_history]
        ),
        "cumulative_length_history": np.stack(
            [item.cumulative_length for item in diagnostics_history]
        ),
        "normalized_cumulative_length_history": np.stack(
            [item.normalized_cumulative_length for item in diagnostics_history]
        ),
        "v_infinity_history": np.stack(
            [item.v_infinity for item in diagnostics_history]
        ),
        "diffusion_time_history": np.stack(
            [item.diffusion_time for item in diagnostics_history]
        ),
        "guidance_weight_history": np.stack(
            [item.guidance_weight for item in diagnostics_history]
        ),
        "sigma_history": np.stack([item.sigma for item in diagnostics_history]),
        "v_eff_history": np.stack([item.v_eff for item in diagnostics_history]),
        "v_dom_history": np.stack([item.v_dom for item in diagnostics_history]),
    }
    for name, values in array_histories.items():
        np.save(run_directory / f"{name}.npy", values)

    np.save(
        run_directory / "total_energy_history.npy",
        np.asarray(
            [item.total_energy for item in diagnostics_history],
            dtype=np.float64,
        ),
    )
    np.save(
        run_directory / "total_length_history.npy",
        np.asarray(
            [item.total_length for item in diagnostics_history],
            dtype=np.float64,
        ),
    )
