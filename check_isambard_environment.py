#!/usr/bin/env python3
"""Verify the Isambard GPU environment and repository inputs."""

import argparse
import importlib.metadata
import os
from pathlib import Path

import torch as th


PACKAGE_NAMES = (
    "blobfile",
    "ehtim",
    "matplotlib",
    "mpi4py",
    "numpy",
    "Pillow",
    "PyYAML",
    "scipy",
    "torch",
    "torchkbnufft",
    "torchvision",
    "wandb",
)


def create_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository",
        default=str(Path(__file__).resolve().parent),
    )
    parser.add_argument(
        "--wandb_mode",
        choices=("online", "offline", "disabled"),
        default="online",
    )
    parser.add_argument("--wandb_project", default="bh_sampling")
    parser.add_argument("--wandb_entity", default="")
    return parser


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required repository file is missing: {path}")
    print(f"[ok] file: {path}")


def check_packages() -> dict[str, str]:
    versions = {}
    for package_name in PACKAGE_NAMES:
        version = importlib.metadata.version(package_name)
        versions[package_name] = version
        print(f"[ok] package: {package_name} {version}")
    return versions


def check_cuda_and_nufft() -> dict[str, object]:
    import torchkbnufft as tkbn

    if not th.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the Isambard setup job.")
    device = th.device("cuda")
    gpu_name = th.cuda.get_device_name(device)
    print(f"[ok] CUDA: {gpu_name}")

    operator = tkbn.KbNufft(im_size=(8, 8)).to(device)
    image = th.randn(
        1,
        1,
        8,
        8,
        dtype=th.complex64,
        device=device,
    )
    trajectory = th.zeros(2, 4, device=device)
    result = operator(image, trajectory)
    if result.shape != (1, 1, 4) or not th.isfinite(result).all():
        raise RuntimeError("The torchkbnufft CUDA smoke test failed.")
    print(f"[ok] torchkbnufft CUDA output: {tuple(result.shape)}")
    return {
        "gpu_name": gpu_name,
        "cuda_runtime": th.version.cuda,
        "cuda_device_count": th.cuda.device_count(),
    }


def check_model(repository: Path) -> None:
    from conditional_diffusion import DiffusionGuidance

    guidance = DiffusionGuidance(th.device("cuda"), img_size=128)
    noisy_images = th.randn(1, 1, 128, 128, device="cuda")
    schedule_values = th.full((1,), 0.5, device="cuda")
    with th.no_grad():
        score = guidance.compute_prior_score(noisy_images, schedule_values)
    if score.shape != noisy_images.shape or not th.isfinite(score).all():
        raise RuntimeError("The diffusion-prior model smoke test failed.")
    print(f"[ok] diffusion prior output: {tuple(score.shape)}")
    del guidance, noisy_images, schedule_values, score
    th.cuda.empty_cache()


def check_likelihoods(repository: Path) -> None:
    from likelihoods import load_log_likelihood

    for data_label in ("hi", "lo"):
        uvfile = (
            repository
            / "bh_util"
            / "sim_files"
            / f"SR1_M87_2017_095_{data_label}_hops_netcal_StokesI.uvfits"
        )
        likelihood = load_log_likelihood(
            "interferometric",
            uvfile=str(uvfile),
            img_size=128,
            device=th.device("cuda"),
            psize=7.5752137673365e-12,
        )
        image = th.rand(
            1,
            1,
            128,
            128,
            device="cuda",
            requires_grad=True,
        )
        value = likelihood(image)
        gradient = th.autograd.grad(value.sum(), image)[0]
        if value.shape != (1,) or not th.isfinite(value).all():
            raise RuntimeError(f"The {data_label} likelihood is non-finite.")
        if gradient.shape != image.shape or not th.isfinite(gradient).all():
            raise RuntimeError(
                f"The {data_label} likelihood gradient is non-finite."
            )
        print(
            f"[ok] EHT {data_label} likelihood and gradient: "
            f"value={float(value.item()):.6g}"
        )
        del likelihood, image, value, gradient
        th.cuda.empty_cache()


def check_wandb(
    *,
    mode: str,
    project: str,
    entity: str,
    package_versions: dict[str, str],
    cuda_summary: dict[str, object],
) -> None:
    import wandb

    job_id = os.environ.get("SLURM_JOB_ID", "manual")
    run = wandb.init(
        project=project,
        entity=entity or None,
        mode=mode,
        name=f"isambard-environment-check-{job_id}",
        job_type="environment-check",
        tags=["isambard", "environment-check"],
        config={
            "packages": package_versions,
            **cuda_summary,
        },
    )
    run.log(
        {
            "environment_check/success": 1,
            "environment_check/cuda_device_count": th.cuda.device_count(),
        }
    )
    run.summary["environment_check_status"] = "passed"
    run.finish()
    print(f"[ok] W&B {mode} run completed")


def main() -> None:
    args = create_argparser().parse_args()
    repository = Path(args.repository).expanduser().resolve()

    os.environ["MODEL_PATH"] = str(repository / "model" / "model300000.pt")
    os.environ["BETA_PATH"] = str(
        repository / "model" / "ddpm_betas300000.npy"
    )
    os.environ["OMEGA_PATH"] = str(
        repository / "model" / "ddpm_omegas300000.npy"
    )

    required_files = (
        repository / "model" / "model300000.pt",
        repository / "model" / "ddpm_betas300000.npy",
        repository / "model" / "ddpm_omegas300000.npy",
        repository / "bh_util" / "sim_files" / "sbar_out_55_554000.npy",
        repository
        / "bh_util"
        / "sim_files"
        / "SR1_M87_2017_095_hi_hops_netcal_StokesI.uvfits",
        repository
        / "bh_util"
        / "sim_files"
        / "SR1_M87_2017_095_lo_hops_netcal_StokesI.uvfits",
    )
    for required_file in required_files:
        require_file(required_file)

    package_versions = check_packages()
    cuda_summary = check_cuda_and_nufft()
    check_model(repository)
    check_likelihoods(repository)
    check_wandb(
        mode=args.wandb_mode,
        project=args.wandb_project,
        entity=args.wandb_entity,
        package_versions=package_versions,
        cuda_summary=cuda_summary,
    )
    print("All Isambard environment checks passed.")


if __name__ == "__main__":
    main()
