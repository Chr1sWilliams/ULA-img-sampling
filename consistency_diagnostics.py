"""Estimate and persist likelihood-score consistency along a diffusion path."""

from dataclasses import dataclass
import csv
import json
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np
import torch as th


TensorFunction = Callable[[th.Tensor, th.Tensor], th.Tensor]
LikelihoodScoreFunction = Callable[[th.Tensor], th.Tensor]
BatchFactory = Callable[[int], Iterable[th.Tensor]]


@dataclass(frozen=True)
class ConsistencyPath:
    """Recorded relative likelihood-score error across an increasing s-grid."""

    schedule: np.ndarray
    eta: np.ndarray
    eta_squared: np.ndarray
    numerator: np.ndarray
    denominator: np.ndarray
    reconstruction_mse: np.ndarray
    clean_score_rms: np.ndarray
    reconstructed_score_rms: np.ndarray
    num_samples: np.ndarray
    tolerance: float
    selected_index: Optional[int]
    selected_schedule_value: Optional[float]


def validate_schedule(schedule: np.ndarray) -> np.ndarray:
    """Return a validated, strictly increasing schedule in (0, 1)."""
    schedule = np.asarray(schedule, dtype=np.float64).reshape(-1)
    if schedule.size == 0:
        raise ValueError("schedule must contain at least one value.")
    if not np.all(np.isfinite(schedule)):
        raise ValueError("schedule contains non-finite values.")
    if np.any(schedule <= 0.0) or np.any(schedule >= 1.0):
        raise ValueError("schedule values must lie strictly between zero and one.")
    if np.any(np.diff(schedule) <= 0.0):
        raise ValueError("schedule must be strictly increasing.")
    return schedule


def forward_diffusion_sample(
    clean_images: th.Tensor,
    schedule_value: float,
    noise: th.Tensor,
) -> th.Tensor:
    """Draw X_s from the variance-preserving forward kernel K_s(.|X_0)."""
    if clean_images.shape != noise.shape:
        raise ValueError("clean_images and noise must have identical shapes.")
    if not 0.0 < schedule_value < 1.0:
        raise ValueError("schedule_value must lie strictly between zero and one.")
    signal_scale = 1.0 - float(schedule_value)
    noise_scale = np.sqrt(1.0 - signal_scale**2)
    return signal_scale * clean_images + noise_scale * noise


def tweedie_reconstruction(
    noisy_images: th.Tensor,
    schedule_value: float,
    prior_score: th.Tensor,
) -> th.Tensor:
    """Apply the Tweedie posterior-mean formula at one diffusion level."""
    if noisy_images.shape != prior_score.shape:
        raise ValueError("noisy_images and prior_score must have identical shapes.")
    if not 0.0 < schedule_value < 1.0:
        raise ValueError("schedule_value must lie strictly between zero and one.")
    signal_scale = 1.0 - float(schedule_value)
    noise_variance = 1.0 - signal_scale**2
    return (noisy_images + noise_variance * prior_score) / signal_scale


def select_consistency_threshold(
    schedule: np.ndarray,
    eta: np.ndarray,
    tolerance: float,
) -> tuple[Optional[int], Optional[float]]:
    """Select the first acceptable point when scanning high noise to low noise.

    Schedules are stored in increasing order, so this is the largest schedule
    value whose recorded relative error is no greater than ``tolerance``.
    """
    schedule = validate_schedule(schedule)
    eta = np.asarray(eta, dtype=np.float64).reshape(-1)
    if eta.shape != schedule.shape:
        raise ValueError("eta and schedule must have identical shapes.")
    if tolerance < 0.0 or not np.isfinite(tolerance):
        raise ValueError("tolerance must be finite and non-negative.")

    for index in range(schedule.size - 1, -1, -1):
        if np.isfinite(eta[index]) and eta[index] <= tolerance:
            return index, float(schedule[index])
    return None, None


def load_selected_threshold(result_path: str) -> float:
    """Load and validate a selected threshold from estimator artifacts."""
    path = Path(result_path).expanduser()
    if path.is_dir():
        path = path / "summary.json"
    if not path.is_file():
        raise FileNotFoundError(f"Consistency result does not exist: {path}")

    if path.suffix.lower() == ".npy":
        value = float(np.asarray(np.load(path)).reshape(()))
    else:
        payload = json.loads(path.read_text())
        value = payload.get("selected_schedule_value")
        if value is None:
            raise ValueError(
                f"Consistency result {path} has no selected threshold."
            )
        value = float(value)
    if not 0.0 < value < 1.0:
        raise ValueError(
            "Resolved guidance threshold must lie between zero and one, "
            f"got {value}."
        )
    return value


def estimate_consistency_path(
    *,
    schedule: np.ndarray,
    clean_batch_factory: BatchFactory,
    prior_score: TensorFunction,
    likelihood_score: LikelihoodScoreFunction,
    tolerance: float,
    device: th.device,
    seed: int = 0,
) -> ConsistencyPath:
    """Estimate eta(s) at every grid point using independent forward draws."""
    schedule = validate_schedule(schedule)
    if tolerance < 0.0 or not np.isfinite(tolerance):
        raise ValueError("tolerance must be finite and non-negative.")

    count = schedule.size
    numerator = np.zeros(count, dtype=np.float64)
    denominator = np.zeros(count, dtype=np.float64)
    reconstruction_mse = np.zeros(count, dtype=np.float64)
    clean_score_squared = np.zeros(count, dtype=np.float64)
    reconstructed_score_squared = np.zeros(count, dtype=np.float64)
    num_samples = np.zeros(count, dtype=np.int64)
    num_elements = np.zeros(count, dtype=np.int64)

    for level_index, schedule_value in enumerate(schedule):
        noise_generator = th.Generator(device=device)
        noise_generator.manual_seed(seed + level_index)

        for clean_batch in clean_batch_factory(level_index):
            clean_batch = clean_batch.to(device=device, dtype=th.get_default_dtype())
            if clean_batch.ndim != 4 or clean_batch.shape[0] == 0:
                raise ValueError(
                    "clean batches must have non-empty shape (B, C, H, W)."
                )
            noise = th.randn(
                clean_batch.shape,
                generator=noise_generator,
                device=device,
                dtype=clean_batch.dtype,
            )
            noisy_batch = forward_diffusion_sample(
                clean_batch,
                float(schedule_value),
                noise,
            )
            schedule_batch = th.full(
                (clean_batch.shape[0],),
                float(schedule_value),
                device=device,
                dtype=clean_batch.dtype,
            )

            with th.no_grad():
                score = prior_score(noisy_batch, schedule_batch)
                reconstruction = tweedie_reconstruction(
                    noisy_batch,
                    float(schedule_value),
                    score,
                )

            clean_likelihood_score = likelihood_score(clean_batch)
            reconstructed_likelihood_score = likelihood_score(reconstruction)
            if clean_likelihood_score.shape != clean_batch.shape:
                raise ValueError(
                    "likelihood_score must return a tensor with the input shape."
                )
            if reconstructed_likelihood_score.shape != reconstruction.shape:
                raise ValueError(
                    "likelihood_score must return a tensor with the input shape."
                )

            difference = (
                clean_likelihood_score - reconstructed_likelihood_score
            )
            numerator[level_index] += float(difference.square().sum().item())
            clean_squared = float(clean_likelihood_score.square().sum().item())
            reconstructed_squared = float(
                reconstructed_likelihood_score.square().sum().item()
            )
            denominator[level_index] += clean_squared
            clean_score_squared[level_index] += clean_squared
            reconstructed_score_squared[level_index] += reconstructed_squared
            reconstruction_mse[level_index] += float(
                (clean_batch - reconstruction).square().sum().item()
            )
            num_samples[level_index] += clean_batch.shape[0]
            num_elements[level_index] += clean_batch.numel()

        if num_samples[level_index] == 0:
            raise ValueError(
                f"clean_batch_factory produced no samples for level {level_index}."
            )
        if denominator[level_index] <= 0.0:
            raise ValueError(
                "The clean likelihood-score energy is zero at schedule value "
                f"{schedule_value:.6g}; use a non-constant likelihood."
            )

    eta_squared = numerator / denominator
    eta = np.sqrt(eta_squared)
    reconstruction_mse = reconstruction_mse / num_elements
    clean_score_rms = np.sqrt(clean_score_squared / num_elements)
    reconstructed_score_rms = np.sqrt(
        reconstructed_score_squared / num_elements
    )
    selected_index, selected_schedule_value = select_consistency_threshold(
        schedule,
        eta,
        tolerance,
    )
    return ConsistencyPath(
        schedule=schedule,
        eta=eta,
        eta_squared=eta_squared,
        numerator=numerator,
        denominator=denominator,
        reconstruction_mse=reconstruction_mse,
        clean_score_rms=clean_score_rms,
        reconstructed_score_rms=reconstructed_score_rms,
        num_samples=num_samples,
        tolerance=float(tolerance),
        selected_index=selected_index,
        selected_schedule_value=selected_schedule_value,
    )


def save_consistency_path(
    result: ConsistencyPath,
    output_directory: Path,
    *,
    metadata: Optional[dict] = None,
) -> None:
    """Save arrays, a CSV path, summary JSON, and a diagnostic plot."""
    output_directory.mkdir(parents=True, exist_ok=True)
    metadata = {} if metadata is None else dict(metadata)

    np.savez_compressed(
        output_directory / "consistency_path.npz",
        schedule=result.schedule,
        eta=result.eta,
        eta_squared=result.eta_squared,
        numerator=result.numerator,
        denominator=result.denominator,
        reconstruction_mse=result.reconstruction_mse,
        clean_score_rms=result.clean_score_rms,
        reconstructed_score_rms=result.reconstructed_score_rms,
        num_samples=result.num_samples,
        tolerance=np.asarray(result.tolerance),
    )

    with (output_directory / "consistency_path.csv").open(
        "w",
        newline="",
    ) as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "grid_index",
                "schedule_value",
                "eta",
                "eta_squared",
                "numerator",
                "denominator",
                "reconstruction_mse",
                "clean_score_rms",
                "reconstructed_score_rms",
                "num_samples",
            ]
        )
        for index in range(result.schedule.size):
            writer.writerow(
                [
                    index,
                    result.schedule[index],
                    result.eta[index],
                    result.eta_squared[index],
                    result.numerator[index],
                    result.denominator[index],
                    result.reconstruction_mse[index],
                    result.clean_score_rms[index],
                    result.reconstructed_score_rms[index],
                    result.num_samples[index],
                ]
            )

    summary = {
        **metadata,
        "tolerance": result.tolerance,
        "selected_index": result.selected_index,
        "selected_schedule_value": result.selected_schedule_value,
        "search_direction": "high_noise_to_low_noise",
        "num_grid_points": int(result.schedule.size),
    }
    (output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n"
    )
    if result.selected_schedule_value is not None:
        np.save(
            output_directory / "guidance_threshold.npy",
            np.asarray(result.selected_schedule_value, dtype=np.float64),
        )

    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7.0, 4.5))
    axis.plot(result.schedule, result.eta, marker="o", markersize=3, label="eta(s)")
    axis.axhline(
        result.tolerance,
        color="tab:red",
        linestyle="--",
        label="tolerance",
    )
    if result.selected_schedule_value is not None:
        axis.axvline(
            result.selected_schedule_value,
            color="tab:green",
            linestyle=":",
            label=f"selected s={result.selected_schedule_value:.4g}",
        )
    axis.set_xlabel("schedule value s")
    axis.set_ylabel("relative likelihood-score error eta")
    axis.set_title("Likelihood-score consistency along unconditional diffusion")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_directory / "consistency_path.png", dpi=180)
    plt.close(figure)
