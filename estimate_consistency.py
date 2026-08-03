#!/usr/bin/env python3
"""Estimate likelihood-score consistency across the unconditional diffusion path."""

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
import torch as th
from torch.utils.data import Dataset

from conditional_diffusion import DiffusionGuidance
from consistency_diagnostics import (
    estimate_consistency_path,
    save_consistency_path,
)
from likelihoods import load_log_likelihood
from util import load_schedule


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif"}


class CleanImageDataset(Dataset):
    """Load q_0 samples with the same grayscale normalization as the prior."""

    def __init__(self, data_dir: str, image_size: int) -> None:
        self.image_size = int(image_size)
        if self.image_size <= 0:
            raise ValueError("image_size must be positive.")
        root = Path(data_dir).expanduser()
        if not root.is_dir():
            raise NotADirectoryError(f"data_dir is not a directory: {root}")
        self.paths = sorted(
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.paths:
            raise ValueError(f"No supported images were found below {root}.")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> th.Tensor:
        with Image.open(self.paths[index]) as image:
            image = image.convert("L")
            while min(image.size) >= 2 * self.image_size:
                image = image.resize(
                    tuple(size // 2 for size in image.size),
                    resample=Image.BOX,
                )
            scale = self.image_size / min(image.size)
            image = image.resize(
                tuple(round(size * scale) for size in image.size),
                resample=Image.BICUBIC,
            )
            array = np.asarray(image, dtype=np.float32)

        crop_y = (array.shape[0] - self.image_size) // 2
        crop_x = (array.shape[1] - self.image_size) // 2
        array = array[
            crop_y : crop_y + self.image_size,
            crop_x : crop_x + self.image_size,
        ]
        array = array / 127.5 - 1.0
        return th.from_numpy(array.copy()).unsqueeze(0)


def create_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_dir",
        required=True,
        help="Directory containing representative clean q_0 images.",
    )
    parser.add_argument(
        "--schedule_path",
        default="bh_util/sim_files/sbar_out_55_554000.npy",
    )
    parser.add_argument("--num_grid_points", type=int, default=50)
    parser.add_argument("--num_samples", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=1)
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
        default="outputs/consistency",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    for name in ("num_grid_points", "num_samples", "batch_size", "image_size"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive.")
    if args.num_grid_points < 2:
        raise ValueError("num_grid_points must be at least two.")
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


def make_batch_factory(
    dataset: CleanImageDataset,
    *,
    num_samples: int,
    batch_size: int,
    seed: int,
    schedule: np.ndarray,
):
    """Create independent-with-replacement q_0 draws for each grid point."""

    def batches(level_index: int) -> Iterable[th.Tensor]:
        print(
            f"[{level_index + 1}/{schedule.size}] "
            f"s={schedule[level_index]:.6g}"
        )
        generator = np.random.default_rng(seed + 100_000 + level_index)
        indices = generator.integers(
            low=0,
            high=len(dataset),
            size=num_samples,
        )
        for start in range(0, num_samples, batch_size):
            batch_indices = indices[start : start + batch_size]
            yield th.stack([dataset[int(index)] for index in batch_indices])

    return batches


def main() -> None:
    args = create_argparser().parse_args()
    validate_args(args)
    device = select_device(args.device)
    output_directory = Path(args.output_dir).expanduser()
    output_directory.mkdir(parents=True, exist_ok=True)

    dataset = CleanImageDataset(args.data_dir, args.image_size)
    schedule = load_schedule(args.schedule_path, args.num_grid_points)
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
    batch_factory = make_batch_factory(
        dataset,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        seed=args.seed,
        schedule=schedule,
    )

    print(f"Using device: {device}")
    print(f"Clean-image population: {len(dataset)} files")
    print(f"Likelihood: {args.log_likelihood}")
    print(f"Recording {schedule.size} grid points in {output_directory}")

    result = estimate_consistency_path(
        schedule=schedule,
        clean_batch_factory=batch_factory,
        prior_score=guidance.compute_prior_score,
        likelihood_score=guidance.compute_likelihood_score,
        tolerance=args.eta_tolerance,
        device=device,
        seed=args.seed,
    )
    metadata = {
        **vars(args),
        "resolved_device": str(device),
        "dataset_size": len(dataset),
        "model_path": os.environ.get("MODEL_PATH", "model/model300000.pt"),
        "beta_path": os.environ.get(
            "BETA_PATH",
            "model/ddpm_betas300000.npy",
        ),
        "omega_path": os.environ.get(
            "OMEGA_PATH",
            "model/ddpm_omegas300000.npy",
        ),
    }
    save_consistency_path(
        result,
        output_directory,
        metadata=metadata,
    )
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
    print(f"Saved consistency path to: {output_directory}")


if __name__ == "__main__":
    main()
