#!/usr/bin/env python3
"""Train a grayscale or RGB diffusion prior for conditional image sampling."""

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch as th

from improved_diffusion import dist_util, logger
from improved_diffusion.image_datasets import load_data
from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.script_util import (
    add_dict_to_argparser,
    args_to_dict,
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)
from improved_diffusion.train_util import TrainLoop


def _load_schedule_vector(path: str, name: str) -> np.ndarray:
    schedule_path = Path(path).expanduser()
    if not schedule_path.is_file():
        raise FileNotFoundError(f"{name} file does not exist: {schedule_path}")

    values = np.asarray(np.load(schedule_path), dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"{name} values must be one-dimensional.")
    if values.size < 2:
        raise ValueError(f"{name} values must contain at least two entries.")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} values must all be finite.")
    return values


def load_custom_schedule(
    beta_values_file: str,
    omega_values_file: str,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Load and validate an optional beta/omega schedule pair."""
    if bool(beta_values_file) != bool(omega_values_file):
        raise ValueError(
            "beta_values_file and omega_values_file must be supplied together."
        )
    if not beta_values_file:
        return None, None

    betas = _load_schedule_vector(beta_values_file, "beta")
    omegas = _load_schedule_vector(omega_values_file, "omega")
    if betas.shape != omegas.shape:
        raise ValueError(
            "beta and omega schedules must have the same number of entries."
        )
    if np.any(betas <= 0.0) or np.any(betas >= 1.0):
        raise ValueError("beta values must lie strictly between zero and one.")
    if np.any(np.diff(omegas) <= 0.0):
        raise ValueError("omega values must be strictly increasing.")
    return betas, omegas


def apply_custom_schedule(
    diffusion,
    betas: Optional[np.ndarray],
    omegas: Optional[np.ndarray],
) -> None:
    """Install a validated custom schedule before constructing its sampler."""
    if betas is None or omegas is None:
        return

    diffusion.betas = betas.copy()
    diffusion.omega = omegas.copy()
    diffusion.num_timesteps = int(betas.size)
    diffusion.beta0 = float(betas[0])
    diffusion.omega_start = float(omegas[0])
    diffusion.omega_end = float(omegas[-1])
    diffusion.lambda_increments = th.zeros(diffusion.num_timesteps)
    diffusion.energy_increments = th.zeros(diffusion.num_timesteps)
    diffusion.Energy = 0.0
    diffusion.Lambda = 0.0
    diffusion.space_error = 0.0
    diffusion.update_alpha()
    diffusion.reset_counters()


def validate_args(args: argparse.Namespace) -> None:
    if not args.data_dir:
        raise ValueError("data_dir is required.")
    if not Path(args.data_dir).expanduser().is_dir():
        raise NotADirectoryError(f"data_dir is not a directory: {args.data_dir}")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if args.microbatch == 0 or args.microbatch < -1:
        raise ValueError("microbatch must be -1 or a positive integer.")
    if args.image_channels not in (1, 3):
        raise ValueError("image_channels must be 1 (grayscale) or 3 (RGB).")
    if args.lr <= 0.0:
        raise ValueError("lr must be positive.")
    if args.schedule_lr <= 0.0:
        raise ValueError("schedule_lr must be positive.")
    if args.save_interval <= 0:
        raise ValueError("save_interval must be positive.")
    if args.log_interval <= 0:
        raise ValueError("log_interval must be positive.")


def create_argparser() -> argparse.ArgumentParser:
    defaults = dict(
        data_dir="",
        output_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        schedule_tune=True,
        schedule_lr=0.01,
        image_channels=1,
        batch_size=1,
        microbatch=-1,
        ema_rate="0.9999",
        log_interval=10,
        save_interval=1000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        beta_values_file="",
        omega_values_file="",
    )
    defaults.update(model_and_diffusion_defaults())
    # These values must remain aligned with DiffusionGuidance.load_model().
    defaults.update(
        image_size=128,
        num_channels=64,
        num_res_blocks=2,
        noise_schedule="cosine",
    )
    parser = argparse.ArgumentParser(
        description="Train a grayscale or RGB diffusion prior used by sample.py."
    )
    add_dict_to_argparser(parser, defaults)
    return parser


def main() -> None:
    args = create_argparser().parse_args()
    validate_args(args)
    betas, omegas = load_custom_schedule(
        args.beta_values_file,
        args.omega_values_file,
    )

    dist_util.setup_dist()
    logger.configure(dir=args.output_dir or None)
    logger.log(f"using device: {dist_util.dev()}")
    logger.log(
        f"image mode: {args.image_channels} channel"
        f"{'s' if args.image_channels != 1 else ''}"
    )
    logger.log(
        "initial schedule: "
        f"{'custom beta/omega arrays' if betas is not None else args.noise_schedule}; "
        f"adaptive learning: {args.schedule_tune}"
    )

    logger.log("creating model and diffusion...")
    model_kwargs = args_to_dict(args, model_and_diffusion_defaults().keys())
    model, diffusion = create_model_and_diffusion(
        **model_kwargs,
        in_channels=args.image_channels,
    )
    apply_custom_schedule(diffusion, betas, omegas)
    if betas is not None:
        logger.log(
            f"loaded {betas.size} diffusion steps from "
            f"{args.beta_values_file} and {args.omega_values_file}"
        )

    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(
        args.schedule_sampler,
        diffusion,
    )

    logger.log("creating data loader...")
    data = load_data(
        data_dir=str(Path(args.data_dir).expanduser()),
        batch_size=args.batch_size,
        image_size=args.image_size,
        image_channels=args.image_channels,
        class_cond=args.class_cond,
    )

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        schedule_tune=args.schedule_tune,
        schedule_lr=args.schedule_lr,
        image_size=args.image_size,
        image_channels=args.image_channels,
    ).run_loop()


if __name__ == "__main__":
    main()
