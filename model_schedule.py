"""Estimate and apply the trainer's diffusion-time schedule objective."""

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import PchipInterpolator
import torch as th


@dataclass(frozen=True)
class ModelScheduleEstimate:
    """Score-increment objective and its equal-length schedule update."""

    original_omega: np.ndarray
    original_betas: np.ndarray
    original_alpha_bar: np.ndarray
    optimized_omega: np.ndarray
    optimized_betas: np.ndarray
    optimized_alpha_bar: np.ndarray
    energy_increments: np.ndarray
    length_increments: np.ndarray
    cumulative_length: np.ndarray
    normalized_cumulative_length: np.ndarray
    num_samples: int
    update_rate: float


def install_diffusion_schedule(
    diffusion: Any,
    *,
    omega: np.ndarray,
    betas: np.ndarray,
) -> None:
    """Install matching model-time and beta arrays on a diffusion object.

    The base diffusion is constructed before the learned schedule is loaded,
    so its original ``num_timesteps`` and bookkeeping tensors may have a
    different length. Keep those fields synchronized with the installed
    arrays before reverse-SDE sampling starts.
    """
    omega = np.asarray(omega, dtype=np.float64).reshape(-1)
    betas = np.asarray(betas, dtype=np.float64).reshape(-1)
    if omega.size < 2:
        raise ValueError("omega and betas must contain at least two values.")
    if omega.shape != betas.shape:
        raise ValueError("omega and betas must have equal lengths.")
    if not np.all(np.isfinite(omega)) or np.any(omega <= 0.0):
        raise ValueError("omega must contain finite positive values.")
    if np.any(np.diff(omega) <= 0.0):
        raise ValueError("omega must be strictly increasing.")
    if not np.all(np.isfinite(betas)) or np.any(betas <= 0.0):
        raise ValueError("betas must contain finite positive values.")
    if np.any(betas >= 1.0):
        raise ValueError("betas must be strictly less than one.")

    diffusion.omega = omega.copy()
    diffusion.betas = betas.copy()
    diffusion.num_timesteps = int(omega.size)
    diffusion.omega_start = float(omega[0])
    diffusion.omega_end = float(omega[-1])
    diffusion.update_alpha()

    for name in ("lambda_increments", "energy_increments", "D", "D2"):
        current = getattr(diffusion, name, None)
        if isinstance(current, th.Tensor):
            setattr(diffusion, name, current.new_zeros(diffusion.num_timesteps))
    current_n_time = getattr(diffusion, "n_time", None)
    if isinstance(current_n_time, th.Tensor):
        diffusion.n_time = current_n_time.new_zeros(diffusion.num_timesteps)
        diffusion.n_time[0] = 1


def vp_betas_from_omega(omega: np.ndarray) -> np.ndarray:
    """Convert increasing VP integrated-noise times to exact discrete betas."""
    omega = np.asarray(omega, dtype=np.float64).reshape(-1)
    if omega.size < 2:
        raise ValueError("omega must contain at least two values.")
    if not np.all(np.isfinite(omega)) or np.any(omega <= 0.0):
        raise ValueError("omega must contain finite positive values.")
    if np.any(np.diff(omega) <= 0.0):
        raise ValueError("omega must be strictly increasing.")

    delta_omega = np.diff(np.concatenate([np.zeros(1), omega]))
    betas = -np.expm1(-delta_omega)
    if np.any(betas <= 0.0) or np.any(betas >= 1.0):
        raise ValueError("The optimized omega produced invalid VP betas.")
    return betas


def schedule_from_weighted_score_energy(
    *,
    omega: np.ndarray,
    energy_increments: np.ndarray,
    update_rate: float = 1.0,
) -> ModelScheduleEstimate:
    """Equalize trainer-style weighted score length over model time.

    ``energy_increments[t-1]`` represents

    ``E[sigma[t-1]**2 * ||score[t-1](X_t) - score[t](X_t)||_2**2]``.

    The square roots are path-length increments. Their normalized cumulative
    sum is inverted with the same shape-preserving cubic interpolation used by
    the trainer, placing the returned model times at equal cumulative length.
    """
    omega = np.asarray(omega, dtype=np.float64).reshape(-1)
    energy_increments = np.asarray(
        energy_increments,
        dtype=np.float64,
    ).reshape(-1)
    if omega.size < 2:
        raise ValueError("omega must contain at least two values.")
    if not np.all(np.isfinite(omega)) or np.any(omega <= 0.0):
        raise ValueError("omega must contain finite positive values.")
    if np.any(np.diff(omega) <= 0.0):
        raise ValueError("omega must be strictly increasing.")
    if energy_increments.shape != (omega.size - 1,):
        raise ValueError(
            "energy_increments must contain one value per omega interval."
        )
    if not np.all(np.isfinite(energy_increments)):
        raise ValueError("energy_increments contains non-finite values.")
    if np.any(energy_increments <= 0.0):
        raise ValueError(
            "Every score-energy increment must be positive before optimizing "
            "the schedule."
        )
    update_rate = float(update_rate)
    if not 0.0 < update_rate <= 1.0:
        raise ValueError("update_rate must lie in (0, 1].")

    length_increments = np.sqrt(energy_increments)
    cumulative_length = np.concatenate(
        [np.zeros(1, dtype=np.float64), np.cumsum(length_increments)]
    )
    normalized_cumulative_length = cumulative_length / cumulative_length[-1]
    equal_length = np.linspace(0.0, 1.0, omega.size)
    proposal = PchipInterpolator(
        normalized_cumulative_length,
        omega,
    )(equal_length)
    proposal[0] = omega[0]
    proposal[-1] = omega[-1]
    optimized_omega = (1.0 - update_rate) * omega + update_rate * proposal
    optimized_omega[0] = omega[0]
    optimized_omega[-1] = omega[-1]
    if np.any(np.diff(optimized_omega) <= 0.0):
        raise ValueError("Schedule interpolation did not remain strictly increasing.")

    original_betas = vp_betas_from_omega(omega)
    optimized_betas = vp_betas_from_omega(optimized_omega)
    return ModelScheduleEstimate(
        original_omega=omega.copy(),
        original_betas=original_betas,
        original_alpha_bar=np.cumprod(1.0 - original_betas),
        optimized_omega=optimized_omega,
        optimized_betas=optimized_betas,
        optimized_alpha_bar=np.cumprod(1.0 - optimized_betas),
        energy_increments=energy_increments.copy(),
        length_increments=length_increments,
        cumulative_length=cumulative_length,
        normalized_cumulative_length=normalized_cumulative_length,
        num_samples=0,
        update_rate=update_rate,
    )


def estimate_model_schedule(
    *,
    model: th.nn.Module,
    diffusion: Any,
    clean_samples: th.Tensor,
    batch_size: int,
    device: th.device,
    seed: int,
    update_rate: float = 1.0,
) -> ModelScheduleEstimate:
    """Estimate the active trainer schedule objective from clean samples."""
    if clean_samples.ndim != 4 or clean_samples.shape[0] == 0:
        raise ValueError("clean_samples must have non-empty shape (N, C, H, W).")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    omega = np.asarray(diffusion.omega, dtype=np.float64).reshape(-1)
    alpha_bar = np.asarray(
        diffusion.alphas_cumprod,
        dtype=np.float64,
    ).reshape(-1)
    if omega.shape != alpha_bar.shape:
        raise ValueError("diffusion omega and alpha_bar must have equal lengths.")
    if omega.size < 2 or np.any(np.diff(omega) <= 0.0):
        raise ValueError("diffusion omega must be strictly increasing.")
    sigma = np.sqrt(1.0 - alpha_bar)
    if np.any(sigma <= 0.0) or not np.all(np.isfinite(sigma)):
        raise ValueError("diffusion noise scales must be finite and positive.")

    model_dtype = next(model.parameters()).dtype
    energy_sum = np.zeros(omega.size - 1, dtype=np.float64)
    sample_count = np.zeros(omega.size - 1, dtype=np.int64)

    model.eval()
    for time_index in range(1, omega.size):
        print(
            f"Schedule objective [{time_index}/{omega.size - 1}] "
            f"omega={omega[time_index]:.6g}"
        )
        generator = th.Generator(device=device)
        generator.manual_seed(int(seed) + 10_000 + time_index)
        sqrt_alpha = float(np.sqrt(alpha_bar[time_index]))
        sigma_current = float(sigma[time_index])
        sigma_previous = float(sigma[time_index - 1])

        for start in range(0, clean_samples.shape[0], batch_size):
            clean_batch = clean_samples[start : start + batch_size].to(
                device=device,
                dtype=model_dtype,
            )
            noise = th.randn(
                clean_batch.shape,
                generator=generator,
                device=device,
                dtype=model_dtype,
            )
            noisy_batch = sqrt_alpha * clean_batch + sigma_current * noise
            current_time = th.full(
                (clean_batch.shape[0],),
                float(omega[time_index]),
                device=device,
                dtype=model_dtype,
            )
            previous_time = th.full(
                (clean_batch.shape[0],),
                float(omega[time_index - 1]),
                device=device,
                dtype=model_dtype,
            )
            with th.no_grad():
                current_score = -model(noisy_batch, current_time) / sigma_current
                previous_score = (
                    -model(noisy_batch, previous_time) / sigma_previous
                )
                per_sample_energy = sigma_previous**2 * (
                    previous_score - current_score
                ).reshape(clean_batch.shape[0], -1).square().sum(dim=1)

            energy_sum[time_index - 1] += float(
                per_sample_energy.double().sum().item()
            )
            sample_count[time_index - 1] += clean_batch.shape[0]

    if np.any(sample_count != clean_samples.shape[0]):
        raise RuntimeError("Not every schedule interval used every clean sample.")
    estimate = schedule_from_weighted_score_energy(
        omega=omega,
        energy_increments=energy_sum / sample_count,
        update_rate=update_rate,
    )
    return replace(
        estimate,
        original_betas=np.asarray(diffusion.betas, dtype=np.float64).copy(),
        original_alpha_bar=alpha_bar.copy(),
        num_samples=int(clean_samples.shape[0]),
    )


def apply_model_schedule(diffusion: Any, estimate: ModelScheduleEstimate) -> None:
    """Install an optimized model-time schedule for reverse-SDE sampling."""
    install_diffusion_schedule(
        diffusion,
        omega=estimate.optimized_omega,
        betas=estimate.optimized_betas,
    )


def generate_reverse_sde_samples(
    *,
    model: th.nn.Module,
    diffusion: Any,
    num_samples: int,
    batch_size: int,
    image_size: int,
    device: th.device,
    seed: int,
) -> th.Tensor:
    """Generate unconditional samples with ancestral reverse-SDE stepping."""
    if num_samples <= 0 or batch_size <= 0 or image_size <= 0:
        raise ValueError("num_samples, batch_size, and image_size must be positive.")
    th.manual_seed(int(seed))
    if device.type == "cuda":
        th.cuda.manual_seed_all(int(seed))

    batches = []
    model.eval()
    for start in range(0, num_samples, batch_size):
        count = min(batch_size, num_samples - start)
        print(
            f"Reverse-SDE samples {start + 1}-{start + count}/{num_samples}"
        )
        with th.no_grad():
            batch = diffusion.p_sample_loop(
                model,
                (count, 1, image_size, image_size),
                clip_denoised=True,
                device=device,
                progress=False,
            )
        batches.append(batch.detach().cpu())
    return th.cat(batches, dim=0)


def omega_to_schedule(omega: np.ndarray) -> np.ndarray:
    """Map VP integrated-noise time to the repository's schedule coordinate."""
    omega = np.asarray(omega, dtype=np.float64)
    return -np.expm1(-0.5 * omega)


def save_model_schedule_estimate(
    estimate: ModelScheduleEstimate,
    output_directory: Path,
) -> None:
    """Persist the full score objective and optimized schedule diagnostics."""
    output_directory.mkdir(parents=True, exist_ok=True)
    optimized_schedule = omega_to_schedule(estimate.optimized_omega)
    arrays = {
        "original_omega": estimate.original_omega,
        "original_betas": estimate.original_betas,
        "original_alpha_bar": estimate.original_alpha_bar,
        "optimized_omega": estimate.optimized_omega,
        "optimized_betas": estimate.optimized_betas,
        "optimized_alpha_bar": estimate.optimized_alpha_bar,
        "optimized_schedule": optimized_schedule,
        "energy_increments": estimate.energy_increments,
        "length_increments": estimate.length_increments,
        "cumulative_length": estimate.cumulative_length,
        "normalized_cumulative_length": estimate.normalized_cumulative_length,
    }
    for name, values in arrays.items():
        np.save(output_directory / f"{name}.npy", values)
    np.savez_compressed(
        output_directory / "model_schedule_estimate.npz",
        **arrays,
        num_samples=np.asarray(estimate.num_samples),
        update_rate=np.asarray(estimate.update_rate),
    )

    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    node = np.arange(estimate.original_omega.size)
    interval = np.arange(1, estimate.original_omega.size)
    axes[0, 0].plot(node, estimate.original_omega, label="original")
    axes[0, 0].plot(node, estimate.optimized_omega, label="optimized")
    axes[0, 0].set_title("Model-time schedule")
    axes[0, 0].set_ylabel("omega")
    axes[0, 0].legend()
    axes[0, 1].plot(interval, estimate.energy_increments)
    axes[0, 1].set_title("Sigma-squared weighted score energy")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_ylabel("D_t")
    axes[1, 0].plot(interval, estimate.length_increments)
    axes[1, 0].set_title("Length increments")
    axes[1, 0].set_ylabel("sqrt(D_t)")
    axes[1, 1].plot(node, estimate.normalized_cumulative_length)
    axes[1, 1].plot(
        node,
        np.linspace(0.0, 1.0, node.size),
        linestyle="--",
        label="equal-length target",
    )
    axes[1, 1].set_title("Normalized cumulative length")
    axes[1, 1].legend()
    for axis in axes.flat:
        axis.set_xlabel("schedule interval/node")
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_directory / "model_schedule_estimate.png", dpi=180)
    plt.close(figure)
