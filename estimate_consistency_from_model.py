#!/usr/bin/env python3
"""Estimate consistency using only unconditional samples from the prior model."""

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import torch as th

from conditional_diffusion import DiffusionGuidance
from consistency_diagnostics import (
    estimate_consistency_path,
    save_consistency_path,
)
from likelihoods import load_log_likelihood
from model_schedule import (
    apply_model_schedule,
    estimate_model_schedule,
    generate_reverse_sde_samples,
    omega_to_schedule,
    save_model_schedule_estimate,
)


def create_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument(
        "--num_grid_points",
        type=int,
        default=50,
        help="Number of optimized-schedule points used for consistency.",
    )
    parser.add_argument(
        "--schedule_update_rate",
        type=float,
        default=1.0,
        help=(
            "Blend weight for the equal-length schedule proposal. One applies "
            "the full optimum; smaller values reproduce a trainer-style "
            "partial update."
        ),
    )
    parser.add_argument("--eta_tolerance", type=float, default=0.1)
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--log_likelihood",
        default="interferometric",
        help=(
            "Likelihood to test: interferometric or module:function. "
            "A constant zero likelihood is not identifiable."
        ),
    )
    parser.add_argument(
        "--uvfile",
        default="bh_util/sim_files/simdata_3598_grmhd5_seed3.uvfits",
    )
    parser.add_argument(
        "--pixel_size",
        "--psize",
        dest="pixel_size",
        type=float,
        default=7.5752137673365e-12,
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/model_consistency",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    for name in ("num_samples", "batch_size", "image_size"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive.")
    if args.num_grid_points < 2:
        raise ValueError("num_grid_points must be at least two.")
    if not 0.0 < args.schedule_update_rate <= 1.0:
        raise ValueError("schedule_update_rate must lie in (0, 1].")
    if args.eta_tolerance < 0.0 or not np.isfinite(args.eta_tolerance):
        raise ValueError("eta_tolerance must be finite and non-negative.")
    if args.log_likelihood.strip().lower() in {"zero", "none"}:
        raise ValueError(
            "Consistency cannot be estimated with the constant zero likelihood."
        )


def select_device(requested: str) -> th.device:
    if requested == "auto":
        return th.device("cuda" if th.cuda.is_available() else "cpu")
    if requested == "cuda" and not th.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return th.device(requested)


def interpolate_schedule(schedule: np.ndarray, count: int) -> np.ndarray:
    schedule = np.asarray(schedule, dtype=np.float64).reshape(-1)
    if schedule.size == count:
        return schedule.copy()
    return np.interp(
        np.linspace(0.0, 1.0, count),
        np.linspace(0.0, 1.0, schedule.size),
        schedule,
    )


def make_generated_batch_factory(
    samples: th.Tensor,
    *,
    batch_size: int,
    schedule: np.ndarray,
):
    """Reuse every generated q_0 sample at every consistency grid point."""

    def batches(level_index: int) -> Iterable[th.Tensor]:
        print(
            f"Consistency [{level_index + 1}/{schedule.size}] "
            f"s={schedule[level_index]:.6g}"
        )
        for start in range(0, samples.shape[0], batch_size):
            yield samples[start : start + batch_size]

    return batches


def save_sample_artifacts(
    samples: th.Tensor,
    output_directory: Path,
    label: str,
) -> None:
    samples = samples.detach().cpu()
    th.save(samples, output_directory / f"{label}_samples.pt")
    np.save(
        output_directory / f"{label}_sample_mean.npy",
        samples.mean(dim=0).numpy(),
    )
    np.save(
        output_directory / f"{label}_sample_std.npy",
        samples.std(dim=0, unbiased=False).numpy(),
    )

    import matplotlib.pyplot as plt

    count = min(16, samples.shape[0])
    figure, axes = plt.subplots(4, 4, figsize=(10, 10))
    for index, axis in enumerate(axes.flat):
        axis.axis("off")
        if index >= count:
            continue
        axis.imshow(
            samples[index].squeeze().numpy(),
            cmap="gray",
            vmin=-1.0,
            vmax=1.0,
        )
        axis.set_title(f"sample {index}")
    figure.tight_layout()
    figure.savefig(output_directory / f"{label}_sample_preview.png", dpi=150)
    plt.close(figure)


def main() -> None:
    args = create_argparser().parse_args()
    validate_args(args)
    device = select_device(args.device)
    output_directory = Path(args.output_dir).expanduser()
    output_directory.mkdir(parents=True, exist_ok=True)

    log_likelihood = load_log_likelihood(
        args.log_likelihood,
        uvfile=args.uvfile,
        img_size=args.image_size,
        device=device,
        psize=args.pixel_size,
    )
    guidance = DiffusionGuidance(
        device,
        log_likelihood=log_likelihood,
        img_size=args.image_size,
    )

    print(f"Using device: {device}")
    print(f"Likelihood: {args.log_likelihood}")
    print(
        f"Generating {args.num_samples} initial unconditional samples with "
        "the model's loaded reverse-SDE schedule."
    )
    initial_samples = generate_reverse_sde_samples(
        model=guidance.model,
        diffusion=guidance.diffusion,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        image_size=args.image_size,
        device=device,
        seed=args.seed,
    )
    save_sample_artifacts(initial_samples, output_directory, "initial")

    print("Estimating sigma-squared weighted adjacent-score increments.")
    schedule_estimate = estimate_model_schedule(
        model=guidance.model,
        diffusion=guidance.diffusion,
        clean_samples=initial_samples,
        batch_size=args.batch_size,
        device=device,
        seed=args.seed,
        update_rate=args.schedule_update_rate,
    )
    save_model_schedule_estimate(schedule_estimate, output_directory)
    apply_model_schedule(guidance.diffusion, schedule_estimate)

    print(
        f"Generating {args.num_samples} unconditional samples with the "
        "optimized reverse-SDE schedule."
    )
    optimized_samples = generate_reverse_sde_samples(
        model=guidance.model,
        diffusion=guidance.diffusion,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        image_size=args.image_size,
        device=device,
        seed=args.seed + 1_000_000,
    )
    save_sample_artifacts(optimized_samples, output_directory, "optimized")

    optimized_schedule = omega_to_schedule(schedule_estimate.optimized_omega)
    consistency_schedule = interpolate_schedule(
        optimized_schedule,
        args.num_grid_points,
    )
    batch_factory = make_generated_batch_factory(
        optimized_samples,
        batch_size=args.batch_size,
        schedule=consistency_schedule,
    )
    print(
        f"Estimating consistency at {consistency_schedule.size} optimized "
        "schedule points from the generated samples."
    )
    result = estimate_consistency_path(
        schedule=consistency_schedule,
        clean_batch_factory=batch_factory,
        prior_score=guidance.compute_prior_score,
        likelihood_score=guidance.compute_likelihood_score,
        tolerance=args.eta_tolerance,
        device=device,
        seed=args.seed + 2_000_000,
    )

    metadata = {
        **vars(args),
        "clean_sample_source": "optimized_reverse_sde_model_samples",
        "resolved_device": str(device),
        "model_path": os.environ.get("MODEL_PATH", "model/model300000.pt"),
        "beta_path": os.environ.get(
            "BETA_PATH",
            "model/ddpm_betas300000.npy",
        ),
        "omega_path": os.environ.get(
            "OMEGA_PATH",
            "model/ddpm_omegas300000.npy",
        ),
        "schedule_objective": (
            "E[sigma_(t-1)^2 * ||score_(t-1)(X_t) - score_t(X_t)||_2^2]"
        ),
        "schedule_length_increment": "sqrt(schedule_objective)",
        "reverse_beta_conversion": "1 - exp(-(omega_t - omega_(t-1)))",
        "optimized_schedule_path": str(
            output_directory / "optimized_schedule.npy"
        ),
    }
    save_consistency_path(result, output_directory, metadata=metadata)
    (output_directory / "config.json").write_text(
        json.dumps(metadata, indent=2, default=str) + "\n"
    )

    if result.selected_schedule_value is None:
        print(
            "No grid point met the requested tolerance; no guidance threshold "
            "was selected."
        )
    else:
        print(
            "Selected guidance threshold: "
            f"s_eta={result.selected_schedule_value:.8g} "
            f"(eta={result.eta[result.selected_index]:.8g})"
        )
    print(f"Saved model-only consistency run to: {output_directory}")


if __name__ == "__main__":
    main()
