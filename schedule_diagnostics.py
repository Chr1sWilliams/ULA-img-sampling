"""Preprint-aligned schedule diagnostics and artifact persistence."""

from dataclasses import dataclass
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch as th


@dataclass(frozen=True)
class RoundDiagnostics:
    """Diagnostics used to optimize one discretized annealing schedule."""

    schedule_before: np.ndarray
    schedule_after: np.ndarray
    raw_energy_increments: np.ndarray
    energy_increments: np.ndarray
    raw_length_increments: np.ndarray
    length_increments: np.ndarray
    cumulative_length: np.ndarray
    normalized_cumulative_length: np.ndarray
    v_infinity: np.ndarray
    diffusion_time: np.ndarray
    guidance_weight: np.ndarray
    sigma: np.ndarray
    v_eff: np.ndarray
    v_dom: np.ndarray
    raw_total_energy: float
    total_energy: float
    raw_total_length: float
    total_length: float
    length_increment_clip_multiple: float
    length_increment_clip_threshold: float
    num_length_increments_clipped: int


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
    length_increment_clip_multiple: float = np.inf,
) -> RoundDiagnostics:
    """Compute optionally clipped increments and the schedule update.

    A finite clip multiple caps each raw length increment at that multiple of
    the mean raw length increment. The effective energy is the square of the
    clipped length, preserving the energy/length relationship used by the
    schedule update. Raw values remain available in the returned diagnostics.
    """
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
    length_increment_clip_multiple = float(length_increment_clip_multiple)
    if (
        np.isnan(length_increment_clip_multiple)
        or length_increment_clip_multiple <= 0.0
    ):
        raise ValueError(
            "length_increment_clip_multiple must be positive or infinity."
        )

    raw_energy_increments = np.maximum(energy_increments, 0.0)
    raw_length_increments = np.sqrt(raw_energy_increments)
    raw_mean_length = float(raw_length_increments.mean())
    if np.isinf(length_increment_clip_multiple):
        length_increment_clip_threshold = np.inf
    else:
        length_increment_clip_threshold = float(
            length_increment_clip_multiple * raw_mean_length
        )
    length_increments = np.minimum(
        raw_length_increments,
        length_increment_clip_threshold,
    )
    energy_increments = np.square(length_increments)
    num_length_increments_clipped = int(
        np.count_nonzero(length_increments < raw_length_increments)
    )
    cumulative_length = np.concatenate(
        [np.zeros(1, dtype=np.float64), np.cumsum(length_increments)]
    )
    raw_total_energy = float(raw_energy_increments.sum())
    total_energy = float(energy_increments.sum())
    raw_total_length = float(raw_length_increments.sum())
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
        raw_energy_increments=raw_energy_increments,
        energy_increments=energy_increments,
        raw_length_increments=raw_length_increments,
        length_increments=length_increments,
        cumulative_length=cumulative_length,
        normalized_cumulative_length=normalized_cumulative_length,
        v_infinity=v_infinity,
        diffusion_time=diffusion_time,
        guidance_weight=guidance_weight,
        sigma=sigma,
        v_eff=v_eff,
        v_dom=v_dom,
        raw_total_energy=raw_total_energy,
        total_energy=total_energy,
        raw_total_length=raw_total_length,
        total_length=total_length,
        length_increment_clip_multiple=length_increment_clip_multiple,
        length_increment_clip_threshold=length_increment_clip_threshold,
        num_length_increments_clipped=num_length_increments_clipped,
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
        "raw_energy_increments": diagnostics.raw_energy_increments,
        "energy_increments": diagnostics.energy_increments,
        "raw_length_increments": diagnostics.raw_length_increments,
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
        raw_total_energy=np.asarray(diagnostics.raw_total_energy),
        total_energy=np.asarray(diagnostics.total_energy),
        raw_total_length=np.asarray(diagnostics.raw_total_length),
        total_length=np.asarray(diagnostics.total_length),
        length_increment_clip_multiple=np.asarray(
            diagnostics.length_increment_clip_multiple
        ),
        length_increment_clip_threshold=np.asarray(
            diagnostics.length_increment_clip_threshold
        ),
        num_length_increments_clipped=np.asarray(
            diagnostics.num_length_increments_clipped
        ),
    )
    samples_cpu = samples.detach().cpu()
    th.save(samples_cpu, round_directory / "samples.pt")
    np.save(
        round_directory / "sample_mean.npy",
        samples_cpu.mean(dim=0).numpy(),
    )
    np.save(
        round_directory / "sample_std.npy",
        samples_cpu.std(dim=0, unbiased=False).numpy(),
    )
    sample_statistics = {
        "mean": float(samples_cpu.mean().item()),
        "standard_deviation": float(
            samples_cpu.std(unbiased=False).item()
        ),
        "minimum": float(samples_cpu.min().item()),
        "maximum": float(samples_cpu.max().item()),
    }
    save_json(
        round_directory / "sample_statistics.json",
        sample_statistics,
    )
    save_json(
        round_directory / "summary.json",
        {
            **metadata,
            "round_index": round_index,
            "raw_total_energy": diagnostics.raw_total_energy,
            "total_energy": diagnostics.total_energy,
            "raw_total_length": diagnostics.raw_total_length,
            "total_length": diagnostics.total_length,
            "length_increment_clip_multiple": (
                diagnostics.length_increment_clip_multiple
            ),
            "length_increment_clip_threshold": (
                diagnostics.length_increment_clip_threshold
            ),
            "num_length_increments_clipped": (
                diagnostics.num_length_increments_clipped
            ),
            "num_schedule_points": diagnostics.schedule_before.size,
            "num_samples": samples.shape[0],
            "sample_statistics": sample_statistics,
        },
    )
    _save_round_plot(round_directory, diagnostics)
    _save_sample_preview(round_directory, samples_cpu)
    return round_directory


def save_run_history(
    *,
    run_directory: Path,
    schedule_history: Sequence[np.ndarray],
    diagnostics_history: Sequence[RoundDiagnostics],
    round_indices: Optional[Sequence[int]] = None,
) -> None:
    """Save aggregate histories, tables, and progression plots."""
    run_directory.mkdir(parents=True, exist_ok=True)
    schedules = np.stack(schedule_history)
    np.save(run_directory / "schedule_history.npy", schedules)
    np.save(run_directory / "latest_schedule.npy", schedules[-1])

    if not diagnostics_history:
        return
    if len(schedule_history) != len(diagnostics_history) + 1:
        raise ValueError(
            "schedule_history must contain the initial schedule plus one "
            "updated schedule per diagnostic round."
        )
    if round_indices is None:
        round_indices = list(range(len(diagnostics_history)))
    if len(round_indices) != len(diagnostics_history):
        raise ValueError(
            "round_indices must contain one value per diagnostic round."
        )
    round_indices = [int(index) for index in round_indices]

    array_histories = {
        "raw_energy_increment_history": np.stack(
            [item.raw_energy_increments for item in diagnostics_history]
        ),
        "energy_increment_history": np.stack(
            [item.energy_increments for item in diagnostics_history]
        ),
        "raw_length_increment_history": np.stack(
            [item.raw_length_increments for item in diagnostics_history]
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
        run_directory / "raw_total_energy_history.npy",
        np.asarray(
            [item.raw_total_energy for item in diagnostics_history],
            dtype=np.float64,
        ),
    )
    np.save(
        run_directory / "total_energy_history.npy",
        np.asarray(
            [item.total_energy for item in diagnostics_history],
            dtype=np.float64,
        ),
    )
    np.save(
        run_directory / "raw_total_length_history.npy",
        np.asarray(
            [item.raw_total_length for item in diagnostics_history],
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
    raw_total_energy = np.asarray(
        [item.raw_total_energy for item in diagnostics_history],
        dtype=np.float64,
    )
    total_energy = np.asarray(
        [item.total_energy for item in diagnostics_history],
        dtype=np.float64,
    )
    raw_total_length = np.asarray(
        [item.raw_total_length for item in diagnostics_history],
        dtype=np.float64,
    )
    total_length = np.asarray(
        [item.total_length for item in diagnostics_history],
        dtype=np.float64,
    )
    schedule_change = schedules[1:] - schedules[:-1]
    schedule_change_rms = np.sqrt(np.mean(schedule_change**2, axis=1))
    schedule_change_max = np.max(np.abs(schedule_change), axis=1)
    clip_multiple = np.asarray(
        [
            item.length_increment_clip_multiple
            for item in diagnostics_history
        ],
        dtype=np.float64,
    )
    clip_threshold = np.asarray(
        [
            item.length_increment_clip_threshold
            for item in diagnostics_history
        ],
        dtype=np.float64,
    )
    clipped_count = np.asarray(
        [
            item.num_length_increments_clipped
            for item in diagnostics_history
        ],
        dtype=np.int64,
    )

    np.savez_compressed(
        run_directory / "run_history.npz",
        round_index=np.asarray(round_indices, dtype=np.int64),
        schedule_history=schedules,
        raw_total_energy_history=raw_total_energy,
        total_energy_history=total_energy,
        raw_total_length_history=raw_total_length,
        total_length_history=total_length,
        length_increment_clip_multiple_history=clip_multiple,
        length_increment_clip_threshold_history=clip_threshold,
        num_length_increments_clipped_history=clipped_count,
        schedule_change_rms=schedule_change_rms,
        schedule_change_max=schedule_change_max,
        **array_histories,
    )
    with (run_directory / "round_progress.csv").open(
        "w",
        newline="",
    ) as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "round_index",
                "raw_total_energy",
                "total_energy",
                "raw_total_length",
                "total_length",
                "length_increment_clip_multiple",
                "length_increment_clip_threshold",
                "num_length_increments_clipped",
                "schedule_change_rms",
                "schedule_change_max",
            ]
        )
        for offset, round_index in enumerate(round_indices):
            writer.writerow(
                [
                    round_index,
                    raw_total_energy[offset],
                    total_energy[offset],
                    raw_total_length[offset],
                    total_length[offset],
                    clip_multiple[offset],
                    clip_threshold[offset],
                    clipped_count[offset],
                    schedule_change_rms[offset],
                    schedule_change_max[offset],
                ]
            )

    _save_run_progress_plots(
        run_directory=run_directory,
        schedules=schedules,
        round_indices=round_indices,
        raw_total_energy=raw_total_energy,
        total_energy=total_energy,
        raw_total_length=raw_total_length,
        total_length=total_length,
        schedule_change_rms=schedule_change_rms,
        schedule_change_max=schedule_change_max,
    )


def _get_pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save_round_plot(
    round_directory: Path,
    diagnostics: RoundDiagnostics,
) -> None:
    plt = _get_pyplot()
    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    node = np.arange(diagnostics.schedule_before.size)
    interval = np.arange(diagnostics.energy_increments.size)

    axes[0, 0].plot(node, diagnostics.schedule_before, label="before")
    axes[0, 0].plot(node, diagnostics.schedule_after, label="after")
    axes[0, 0].set_title("Schedule update")
    axes[0, 0].set_xlabel("node")
    axes[0, 0].set_ylabel("s")
    axes[0, 0].legend()

    axes[0, 1].plot(
        interval,
        diagnostics.raw_energy_increments,
        label="raw",
        alpha=0.7,
    )
    axes[0, 1].plot(
        interval,
        diagnostics.energy_increments,
        label="used",
    )
    axes[0, 1].set_title("Energy increments")
    axes[0, 1].set_xlabel("interval")
    axes[0, 1].set_ylabel("energy")
    axes[0, 1].legend()

    axes[1, 0].plot(
        interval,
        diagnostics.raw_length_increments,
        label="raw",
        alpha=0.7,
    )
    axes[1, 0].plot(
        interval,
        diagnostics.length_increments,
        label="used",
    )
    axes[1, 0].set_title("Length increments")
    axes[1, 0].set_xlabel("interval")
    axes[1, 0].set_ylabel("length")
    axes[1, 0].legend()

    axes[1, 1].plot(node, diagnostics.normalized_cumulative_length)
    axes[1, 1].plot(
        node,
        np.linspace(0.0, 1.0, node.size),
        linestyle="--",
        label="constant-speed target",
    )
    axes[1, 1].set_title("Normalized cumulative length")
    axes[1, 1].set_xlabel("node")
    axes[1, 1].set_ylabel("normalized length")
    axes[1, 1].legend()

    for axis in axes.flat:
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(round_directory / "round_diagnostics.png", dpi=180)
    plt.close(figure)


def _save_sample_preview(
    round_directory: Path,
    samples: th.Tensor,
    max_samples: int = 16,
) -> None:
    plt = _get_pyplot()
    count = min(max_samples, samples.shape[0])
    columns = min(4, count)
    rows = int(np.ceil(count / columns))
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(3 * columns, 3 * rows),
        squeeze=False,
    )
    for index, axis in enumerate(axes.flat):
        axis.axis("off")
        if index >= count:
            continue
        image = samples[index].squeeze().numpy()
        image_min = float(image.min())
        image_range = float(image.max() - image_min)
        if image_range > 0.0:
            image = (image - image_min) / image_range
        else:
            image = np.zeros_like(image)
        axis.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
        axis.set_title(f"sample {index}")
    figure.tight_layout()
    figure.savefig(round_directory / "sample_preview.png", dpi=150)
    plt.close(figure)


def _save_run_progress_plots(
    *,
    run_directory: Path,
    schedules: np.ndarray,
    round_indices: Sequence[int],
    raw_total_energy: np.ndarray,
    total_energy: np.ndarray,
    raw_total_length: np.ndarray,
    total_length: np.ndarray,
    schedule_change_rms: np.ndarray,
    schedule_change_max: np.ndarray,
) -> None:
    plt = _get_pyplot()
    normalized_node = np.linspace(0.0, 1.0, schedules.shape[1])

    schedule_figure, schedule_axis = plt.subplots(figsize=(8, 5))
    schedule_axis.plot(
        normalized_node,
        schedules[0],
        label="initial",
        linewidth=2,
    )
    for offset, round_index in enumerate(round_indices):
        schedule_axis.plot(
            normalized_node,
            schedules[offset + 1],
            label=f"after round {round_index}",
        )
    schedule_axis.set_xlabel("normalized node")
    schedule_axis.set_ylabel("schedule value s")
    schedule_axis.set_title("Schedule progression across rounds")
    schedule_axis.grid(alpha=0.25)
    schedule_axis.legend()
    schedule_figure.tight_layout()
    schedule_figure.savefig(
        run_directory / "schedule_progression.png",
        dpi=180,
    )
    plt.close(schedule_figure)

    progress_figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes[0, 0].plot(
        round_indices,
        raw_total_energy,
        marker="o",
        label="raw",
    )
    axes[0, 0].plot(
        round_indices,
        total_energy,
        marker="o",
        label="used",
    )
    axes[0, 0].set_title("Total energy")
    axes[0, 0].set_ylabel("energy")
    axes[0, 0].legend()
    axes[0, 1].plot(
        round_indices,
        raw_total_length,
        marker="o",
        label="raw",
    )
    axes[0, 1].plot(
        round_indices,
        total_length,
        marker="o",
        label="used",
    )
    axes[0, 1].set_title("Total length")
    axes[0, 1].set_ylabel("length")
    axes[0, 1].legend()
    axes[1, 0].plot(round_indices, schedule_change_rms, marker="o")
    axes[1, 0].set_title("Schedule RMS change")
    axes[1, 0].set_ylabel("RMS delta s")
    axes[1, 1].plot(round_indices, schedule_change_max, marker="o")
    axes[1, 1].set_title("Schedule maximum change")
    axes[1, 1].set_ylabel("max |delta s|")
    for axis in axes.flat:
        axis.set_xlabel("round")
        axis.grid(alpha=0.25)
    progress_figure.tight_layout()
    progress_figure.savefig(
        run_directory / "energy_length_progression.png",
        dpi=180,
    )
    plt.close(progress_figure)
