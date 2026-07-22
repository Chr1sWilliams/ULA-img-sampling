#!/usr/bin/env python3
"""Optimize an image diffusion schedule with conditional Langevin correctors."""

import argparse
from dataclasses import dataclass
import getpass
from pathlib import Path
from typing import Any

import numpy as np
import torch as th

from conditional_diffusion import DiffusionGuidance
from likelihoods import load_log_likelihood
from schedule_diagnostics import (
    RoundDiagnostics,
    build_round_diagnostics,
    save_json,
    save_round_artifacts,
    save_run_history,
)
from util import load_schedule


@dataclass(frozen=True)
class LangevinSpeedSchedule:
    """Diffusion and Langevin quantities evaluated at every schedule node."""

    diffusion_time: tuple[th.Tensor, ...]
    guidance_weight: tuple[th.Tensor, ...]
    sigma: tuple[th.Tensor, ...]
    v_eff: tuple[th.Tensor, ...]
    v_dom: tuple[th.Tensor, ...]

    @staticmethod
    def _to_numpy(values: tuple[th.Tensor, ...]) -> np.ndarray:
        return np.asarray([float(value.item()) for value in values])

    def numpy_arrays(self) -> dict[str, np.ndarray]:
        return {
            "diffusion_time": self._to_numpy(self.diffusion_time),
            "guidance_weight": self._to_numpy(self.guidance_weight),
            "sigma": self._to_numpy(self.sigma),
            "v_eff": self._to_numpy(self.v_eff),
            "v_dom": self._to_numpy(self.v_dom),
        }


def create_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run preprint-aligned corrector sampling and optimize the annealing "
            "schedule from state-dependent Fisher divergences."
        )
    )
    parser.add_argument(
        "--schedule_path",
        "--s_bar_path",
        dest="schedule_path",
        default="bh_util/sim_files/sbar_out_55_554000.npy",
    )
    parser.add_argument(
        "--num_schedule_points",
        "--N_chains",
        dest="num_schedule_points",
        type=int,
        default=4000,
    )
    parser.add_argument(
        "--num_rounds",
        "--N_rounds",
        dest="num_rounds",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--start_round",
        "--N_rounds_init",
        dest="start_round",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--corrector_steps",
        "--N_correct_init",
        dest="corrector_steps",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--num_samples",
        "--N_sample",
        dest="num_samples",
        type=int,
        default=40,
    )
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--initial_sigma", "--sigma0", type=float, default=0.1)
    parser.add_argument(
        "--step_tail_probability",
        "--alpha_step",
        dest="step_tail_probability",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--log_likelihood",
        default="zero",
        help=(
            "Likelihood to use: interferometric, zero, or an import path "
            "formatted as module:function (default: zero)."
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

    parser.add_argument("--output_dir", "--save_path", default="")
    parser.add_argument("--run_name", default="")
    parser.add_argument("--experiment_id", "--round", default="55")
    parser.add_argument("--task_id", type=int, default=554000)
    parser.add_argument("--slurm_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )

    parser.add_argument("--wandb_project", default="bh_sampling")
    parser.add_argument(
        "--wandb_mode",
        choices=("online", "offline", "disabled"),
        default="online",
    )
    parser.add_argument(
        "--log_interval",
        "--log",
        dest="log_interval",
        type=int,
        default=100,
    )
    return parser


def select_device(requested_device: str) -> th.device:
    if requested_device == "auto":
        return th.device("cuda" if th.cuda.is_available() else "cpu")
    if requested_device == "cuda" and not th.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return th.device(requested_device)


def validate_args(args: argparse.Namespace) -> None:
    """Fail early on invalid run settings."""
    positive_integer_args = {
        "num_schedule_points": args.num_schedule_points,
        "num_rounds": args.num_rounds,
        "num_samples": args.num_samples,
        "image_size": args.image_size,
    }
    for name, value in positive_integer_args.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive.")
    if args.num_schedule_points < 2:
        raise ValueError("num_schedule_points must be at least 2.")
    if args.start_round < 0:
        raise ValueError("start_round must be non-negative.")
    if args.corrector_steps < 0:
        raise ValueError("corrector_steps must be non-negative.")
    if args.initial_sigma <= 0.0:
        raise ValueError("initial_sigma must be positive.")
    if not 0.0 < args.step_tail_probability < 1.0:
        raise ValueError("step_tail_probability must lie strictly between 0 and 1.")
    if args.log_interval < 0:
        raise ValueError("log_interval must be non-negative.")


def build_run_name(args: argparse.Namespace) -> str:
    if args.run_name:
        run_name = args.run_name
    else:
        run_name = (
            f"experiment-{args.experiment_id}_slurm-{args.slurm_id}_"
            f"task-{args.task_id}"
        )
    if Path(run_name).name != run_name:
        raise ValueError("run_name must not contain directory separators.")
    return run_name


def compute_langevin_speed_schedule(
    guidance: DiffusionGuidance,
    schedule: np.ndarray,
    *,
    initial_sigma: float,
    tail_probability: float,
    device: th.device,
    dtype: th.dtype,
) -> LangevinSpeedSchedule:
    schedule_tensor = th.as_tensor(schedule, device=device, dtype=dtype)
    diffusion_times = guidance.diffusion_time(schedule_tensor)
    guidance_weights = guidance.guidance_weight(diffusion_times)
    sigma_schedule = th.sqrt(
        initial_sigma**2 * th.exp(-diffusion_times)
        + 1.0
        - th.exp(-diffusion_times)
    )

    sigma_values: list[th.Tensor] = []
    v_eff_values: list[th.Tensor] = []
    v_dom_values: list[th.Tensor] = []

    for sigma in sigma_schedule.unbind():
        v_eff, v_dom = guidance.compute_langevin_speeds(
            sigma,
            tail_probability=tail_probability,
        )
        sigma_values.append(sigma)
        v_eff_values.append(v_eff)
        v_dom_values.append(v_dom)

    return LangevinSpeedSchedule(
        diffusion_time=tuple(diffusion_times.unbind()),
        guidance_weight=tuple(guidance_weights.unbind()),
        sigma=tuple(sigma_values),
        v_eff=tuple(v_eff_values),
        v_dom=tuple(v_dom_values),
    )


def run_sampling_round(
    *,
    guidance: DiffusionGuidance,
    schedule: np.ndarray,
    speed_schedule: LangevinSpeedSchedule,
    num_samples: int,
    corrector_steps: int,
    log_interval: int,
    seed: int,
    device: th.device,
    dtype: th.dtype,
    wandb_module: Any,
) -> tuple[th.Tensor, np.ndarray, np.ndarray]:
    """Sample one round and estimate every interval energy increment."""
    th.manual_seed(seed)
    if device.type == "cuda":
        th.cuda.manual_seed_all(seed)

    sample_shape = (num_samples, 1, guidance.img_size, guidance.img_size)
    samples = th.randn(
        sample_shape,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    energy_increments = np.zeros(schedule.size - 1, dtype=np.float64)
    v_infinity = np.zeros(schedule.size - 1, dtype=np.float64)

    for node_index in range(schedule.size - 1, -1, -1):
        guidance.step += 1
        current_level = th.full(
            (num_samples,),
            float(schedule[node_index]),
            device=device,
            dtype=dtype,
        )
        guidance_strength = float(
            speed_schedule.guidance_weight[node_index].item()
        )

        samples = guidance.conditional_corrector_step(
            samples,
            schedule_values=current_level,
            num_steps=corrector_steps,
            guidance_strength=guidance_strength,
            v_eff=speed_schedule.v_eff[node_index],
            v_dom=speed_schedule.v_dom[node_index],
        )

        if node_index == 0:
            continue

        previous_level = th.full(
            (num_samples,),
            float(schedule[node_index - 1]),
            device=device,
            dtype=dtype,
        )
        previous_guidance_strength = float(
            speed_schedule.guidance_weight[node_index - 1].item()
        )

        current_score, mean_log_likelihood = guidance.compute_conditional_score(
            samples,
            schedule_values=current_level,
            guidance_strength=guidance_strength,
            return_mean_log_likelihood=True,
        )
        previous_score = guidance.compute_conditional_score(
            samples,
            schedule_values=previous_level,
            guidance_strength=previous_guidance_strength,
            return_mean_log_likelihood=True,
        )[0]
        previous_stepsize = guidance.compute_stepsize(
            previous_score.detach(),
            v_eff=speed_schedule.v_eff[node_index - 1],
            v_dom=speed_schedule.v_dom[node_index - 1],
            eps=1e-12,
        )
        divergence, interval_v_infinity = guidance.compute_divergence(
            current_score.detach(),
            previous_score.detach(),
            previous_stepsize,
        )

        interval_index = node_index - 1
        energy_increments[interval_index] = float(divergence.item())
        v_infinity[interval_index] = float(interval_v_infinity.item())

        if log_interval > 0 and guidance.step % log_interval == 0:
            wandb_module.log(
                {
                    "sampling/node_index": node_index,
                    "sampling/v_infinity": v_infinity[interval_index],
                    "sampling/energy_increment": energy_increments[interval_index],
                    "sampling/log_likelihood": float(mean_log_likelihood),
                    "sampling/score_squared_norm": float(
                        current_score.square()
                        .sum(dim=(1, 2, 3))
                        .mean()
                        .item()
                    ),
                },
                step=guidance.step,
            )

        if device.type == "cuda":
            th.cuda.empty_cache()

    return samples.detach(), energy_increments, v_infinity


def normalize_image(image: th.Tensor) -> th.Tensor:
    image_min = image.min()
    image_range = image.max() - image_min
    if float(image_range) <= 0.0:
        return th.zeros_like(image)
    return (image - image_min) / image_range


def log_round_to_wandb(
    *,
    wandb_module: Any,
    diagnostics: RoundDiagnostics,
    samples: th.Tensor,
    round_index: int,
    step: int,
) -> None:
    schedule_table = wandb_module.Table(
        columns=[
            "node",
            "schedule_before",
            "schedule_after",
            "cumulative_length",
            "normalized_cumulative_length",
            "diffusion_time",
            "guidance_weight",
            "sigma",
            "v_eff",
            "v_dom",
        ]
    )
    for node_index in range(diagnostics.schedule_before.size):
        schedule_table.add_data(
            node_index,
            float(diagnostics.schedule_before[node_index]),
            float(diagnostics.schedule_after[node_index]),
            float(diagnostics.cumulative_length[node_index]),
            float(diagnostics.normalized_cumulative_length[node_index]),
            float(diagnostics.diffusion_time[node_index]),
            float(diagnostics.guidance_weight[node_index]),
            float(diagnostics.sigma[node_index]),
            float(diagnostics.v_eff[node_index]),
            float(diagnostics.v_dom[node_index]),
        )

    increment_table = wandb_module.Table(
        columns=[
            "interval",
            "energy_increment",
            "length_increment",
            "v_infinity",
        ]
    )
    for interval_index in range(diagnostics.energy_increments.size):
        increment_table.add_data(
            interval_index,
            float(diagnostics.energy_increments[interval_index]),
            float(diagnostics.length_increments[interval_index]),
            float(diagnostics.v_infinity[interval_index]),
        )

    samples_cpu = samples.detach().cpu()
    sample_images = []
    for sample_index in range(min(10, samples_cpu.shape[0])):
        image = normalize_image(samples_cpu[sample_index].squeeze())
        sample_images.append(
            wandb_module.Image(image.numpy(), caption=f"sample_{sample_index}")
        )
    mean_image = normalize_image(samples_cpu.mean(dim=0).squeeze())

    wandb_module.log(
        {
            "round/index": round_index,
            "round/total_energy": diagnostics.total_energy,
            "round/total_length": diagnostics.total_length,
            "round/schedule": wandb_module.plot.line_series(
                xs=[
                    np.arange(diagnostics.schedule_before.size),
                    np.arange(diagnostics.schedule_after.size),
                ],
                ys=[
                    diagnostics.schedule_before,
                    diagnostics.schedule_after,
                ],
                keys=["schedule_before", "schedule_after"],
                title=f"Schedule round {round_index}",
                xname="node",
            ),
            "round/cumulative_length": wandb_module.plot.line(
                schedule_table,
                "node",
                "normalized_cumulative_length",
                title=f"Normalized cumulative length round {round_index}",
            ),
            "round/energy_increments": wandb_module.plot.line(
                increment_table,
                "interval",
                "energy_increment",
                title=f"Energy increments round {round_index}",
            ),
            "round/length_increments": wandb_module.plot.line(
                increment_table,
                "interval",
                "length_increment",
                title=f"Length increments round {round_index}",
            ),
            "round/final_samples": sample_images,
            "round/mean_image": wandb_module.Image(
                mean_image.numpy(),
                caption=f"Mean image round {round_index}",
            ),
        },
        step=step,
    )


def main() -> None:
    import wandb

    args = create_argparser().parse_args()
    validate_args(args)
    device = select_device(args.device)
    dtype = th.get_default_dtype()
    run_name = build_run_name(args)
    output_root = (
        Path(args.output_dir)
        if args.output_dir
        else Path("outputs") / getpass.getuser()
    )
    run_directory = output_root / run_name
    run_directory.mkdir(parents=True, exist_ok=True)

    run = wandb.init(
        project=args.wandb_project,
        name=run_name,
        mode=args.wandb_mode,
        config=vars(args),
    )
    save_json(
        run_directory / "config.json",
        {**vars(args), "resolved_device": str(device), "run_name": run_name},
    )

    print(f"Using device: {device}")
    print(f"Using log likelihood: {args.log_likelihood}")
    print(f"Saving run artifacts to: {run_directory}")

    try:
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
        schedule = load_schedule(
            args.schedule_path,
            args.num_schedule_points,
        )

        schedule_history = [schedule.copy()]
        diagnostics_history: list[RoundDiagnostics] = []

        for round_offset in range(args.num_rounds):
            round_index = args.start_round + round_offset
            print(f"\n=== ROUND {round_index} ===")

            speed_schedule = compute_langevin_speed_schedule(
                guidance,
                schedule,
                initial_sigma=args.initial_sigma,
                tail_probability=args.step_tail_probability,
                device=device,
                dtype=dtype,
            )
            samples, energy_increments, v_infinity = run_sampling_round(
                guidance=guidance,
                schedule=schedule,
                speed_schedule=speed_schedule,
                num_samples=args.num_samples,
                corrector_steps=args.corrector_steps,
                log_interval=args.log_interval,
                seed=args.seed + round_index,
                device=device,
                dtype=dtype,
                wandb_module=wandb,
            )
            speed_arrays = speed_schedule.numpy_arrays()
            diagnostics = build_round_diagnostics(
                schedule=schedule,
                energy_increments=energy_increments,
                v_infinity=v_infinity,
                diffusion_time=speed_arrays["diffusion_time"],
                guidance_weight=speed_arrays["guidance_weight"],
                sigma=speed_arrays["sigma"],
                v_eff=speed_arrays["v_eff"],
                v_dom=speed_arrays["v_dom"],
            )

            round_directory = save_round_artifacts(
                run_directory=run_directory,
                round_index=round_index,
                diagnostics=diagnostics,
                samples=samples,
                metadata={
                    "log_likelihood": args.log_likelihood,
                    "seed": args.seed + round_index,
                },
            )
            schedule = diagnostics.schedule_after.copy()
            schedule_history.append(schedule.copy())
            diagnostics_history.append(diagnostics)
            save_run_history(
                run_directory=run_directory,
                schedule_history=schedule_history,
                diagnostics_history=diagnostics_history,
            )
            log_round_to_wandb(
                wandb_module=wandb,
                diagnostics=diagnostics,
                samples=samples,
                round_index=round_index,
                step=guidance.step,
            )

            print(f"Energy: {diagnostics.total_energy:.6g}")
            print(f"Length: {diagnostics.total_length:.6g}")
            print(f"Saved round artifacts to: {round_directory}")
    finally:
        run.finish()


if __name__ == "__main__":
    main()
