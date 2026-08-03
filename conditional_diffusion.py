"""Conditional diffusion guidance and adaptive Langevin updates."""

import os
from typing import Optional, Tuple, Union

import numpy as np
from scipy.stats import chi2
import torch as th

from improved_diffusion import dist_util
from improved_diffusion.script_util import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)
from likelihoods import LogLikelihood, zero_log_likelihood


Scalar = Union[float, th.Tensor]


class DiffusionGuidance:
    """Combine a diffusion prior with optional differentiable data guidance."""

    def __init__(
        self,
        device: th.device,
        log_likelihood: Optional[LogLikelihood] = None,
        img_size: int = 128,
        img_channels: int = 1,
    ) -> None:
        self.img_size = int(img_size)
        if self.img_size <= 0:
            raise ValueError("img_size must be positive.")
        self.img_channels = int(img_channels)
        if self.img_channels not in (1, 3):
            raise ValueError("img_channels must be 1 (grayscale) or 3 (RGB).")
        if log_likelihood is not None and not callable(log_likelihood):
            raise TypeError("log_likelihood must be callable or None.")

        self.log_likelihood = (
            zero_log_likelihood if log_likelihood is None else log_likelihood
        )
        self.model, self.diffusion = self.load_model(device)
        self.step = 0

    @staticmethod
    def alpha(schedule_value: Union[float, np.ndarray, th.Tensor]):
        """Return the variance-preserving signal coefficient ``a(s)``."""
        return (1.0 - schedule_value) ** 2

    @classmethod
    def diffusion_time(
        cls,
        schedule_value: Union[float, np.ndarray, th.Tensor],
    ):
        """Map an annealing schedule value to diffusion time."""
        alpha = cls.alpha(schedule_value)
        if isinstance(alpha, th.Tensor):
            return -th.log(alpha)
        return -np.log(alpha)

    @staticmethod
    def guidance_weight(
        diffusion_time: th.Tensor,
        cutoff: float = 1.0 - np.exp(-0.5),
    ) -> th.Tensor:
        """Return the bounded likelihood-guidance schedule from the preprint."""
        if not 0.0 < cutoff < 1.0:
            raise ValueError("cutoff must lie strictly between zero and one.")
        noising = 1.0 - th.exp(-0.5 * diffusion_time)
        return th.clamp(1.0 - noising / cutoff, min=0.0)

    def load_model(self, device: th.device):
        """Load the pretrained model and its learned diffusion parameters."""
        model_config = model_and_diffusion_defaults()
        model_config.update(
            image_size=self.img_size,
            num_res_blocks=2,
            num_channels=64,
        )
        model, diffusion = create_model_and_diffusion(
            **model_config,
            in_channels=self.img_channels,
        )

        model_path = os.environ.get("MODEL_PATH", "model/model300000.pt")
        beta_path = os.environ.get("BETA_PATH", "model/ddpm_betas300000.npy")
        omega_path = os.environ.get("OMEGA_PATH", "model/ddpm_omegas300000.npy")

        model.load_state_dict(
            dist_util.load_state_dict(model_path, map_location=device)
        )
        model = model.to(device)
        model.eval()

        diffusion.omega = np.load(omega_path)
        diffusion.betas = np.load(beta_path)
        diffusion.omega_start = diffusion.omega[0]
        diffusion.omega_end = diffusion.omega[-1]
        diffusion.update_alpha()
        return model, diffusion

    def compute_prior_score(
        self,
        noisy_images: th.Tensor,
        schedule_values: th.Tensor,
    ) -> th.Tensor:
        """Evaluate the unconditional diffusion-prior score at X_s."""
        if noisy_images.ndim != 4:
            raise ValueError(
                "noisy_images must have shape (B, C, H, W), got "
                f"{tuple(noisy_images.shape)}"
            )
        schedule_values = schedule_values.to(
            device=noisy_images.device,
            dtype=noisy_images.dtype,
        ).reshape(-1)
        if schedule_values.shape != (noisy_images.shape[0],):
            raise ValueError("schedule_values must contain one value per image.")
        if th.any(schedule_values <= 0.0) or th.any(schedule_values >= 1.0):
            raise ValueError("schedule_values must lie strictly between zero and one.")

        signal_coefficient = self.alpha(schedule_values)
        noise_scale = th.sqrt(1.0 - signal_coefficient).view(-1, 1, 1, 1)
        diffusion_times = self.diffusion_time(schedule_values)
        return -self.model(noisy_images, diffusion_times) / noise_scale

    def tweedie_denoise(
        self,
        noisy_images: th.Tensor,
        schedule_values: th.Tensor,
        prior_score: Optional[th.Tensor] = None,
    ) -> th.Tensor:
        """Return the Tweedie clean-image estimate from X_s."""
        schedule_values = schedule_values.to(
            device=noisy_images.device,
            dtype=noisy_images.dtype,
        ).reshape(-1)
        if schedule_values.shape != (noisy_images.shape[0],):
            raise ValueError("schedule_values must contain one value per image.")
        if th.any(schedule_values <= 0.0) or th.any(schedule_values >= 1.0):
            raise ValueError("schedule_values must lie strictly between zero and one.")
        if prior_score is None:
            prior_score = self.compute_prior_score(
                noisy_images,
                schedule_values,
            )
        if prior_score.shape != noisy_images.shape:
            raise ValueError("prior_score must have the same shape as noisy_images.")

        signal_coefficient = self.alpha(schedule_values)
        inverse_signal_scale = th.rsqrt(signal_coefficient).view(-1, 1, 1, 1)
        noise_variance = (1.0 - signal_coefficient).view(-1, 1, 1, 1)
        return inverse_signal_scale * (
            noisy_images + noise_variance * prior_score
        )

    def compute_likelihood_score(self, clean_images: th.Tensor) -> th.Tensor:
        """Evaluate grad_x log L(y|x) in the model's [-1, 1] image coordinates."""
        images = clean_images.detach().requires_grad_(True)
        positive_images = th.clamp(images + 1.0, min=1e-15)
        log_likelihood = self.log_likelihood(positive_images)
        if not isinstance(log_likelihood, th.Tensor):
            raise TypeError("log_likelihood must return a torch.Tensor.")
        expected_shape = (images.shape[0],)
        if log_likelihood.shape != expected_shape:
            raise ValueError(
                "log_likelihood must return one value per sample with shape "
                f"{expected_shape}, got {tuple(log_likelihood.shape)}"
            )
        return th.autograd.grad(
            outputs=log_likelihood,
            inputs=images,
            grad_outputs=th.ones_like(log_likelihood),
        )[0].detach()

    def compute_langevin_speeds(
        self,
        sigma: Scalar,
        tail_probability: float = 0.05,
    ) -> Tuple[th.Tensor, th.Tensor]:
        """Calibrate ``v_eff`` and ``v_dom`` from ``sigma``.

        The marginal-Gaussian calibration is

        ``v_eff = sigma**2`` and
        ``v_dom = v_eff * sqrt(chi2_quantile) / sigma``.
        """
        if not 0.0 < tail_probability < 1.0:
            raise ValueError("tail_probability must lie strictly between 0 and 1.")

        sigma = th.as_tensor(sigma)
        if not sigma.is_floating_point():
            sigma = sigma.to(dtype=th.get_default_dtype())
        if sigma.numel() != 1:
            raise ValueError("sigma must be a scalar value.")
        if sigma.item() <= 0:
            raise ValueError("sigma must be positive.")

        image_dimension = self.img_size**2
        score_radius = float(
            np.sqrt(
                chi2.ppf(
                    1.0 - tail_probability,
                    df=image_dimension,
                )
            )
        )
        v_eff = sigma.square()
        v_dom = v_eff * score_radius / sigma
        return v_eff, v_dom

    @staticmethod
    def compute_stepsize(
        score: th.Tensor,
        v_eff: Scalar,
        v_dom: Scalar,
        eps: float = 1e-5,
    ) -> th.Tensor:
        """Compute the clipped Langevin step size for each image.

        The step size is ``v_eff`` when ``||score|| <= v_dom / v_eff`` and
        ``v_dom / ||score||`` otherwise.
        """
        if score.ndim != 4:
            raise ValueError(
                f"score must have shape (B, C, H, W), got {tuple(score.shape)}"
            )
        if eps <= 0:
            raise ValueError("eps must be positive.")

        v_eff = th.as_tensor(v_eff, dtype=score.dtype, device=score.device)
        v_dom = th.as_tensor(v_dom, dtype=score.dtype, device=score.device)
        if v_eff.numel() != 1 or v_dom.numel() != 1:
            raise ValueError("v_eff and v_dom must be scalar values.")
        if v_eff.item() <= 0:
            raise ValueError("v_eff must be positive.")
        if v_dom.item() <= 0:
            raise ValueError("v_dom must be positive.")

        score_norm = th.linalg.vector_norm(
            score.reshape(score.shape[0], -1),
            dim=1,
            keepdim=True,
        ).view(score.shape[0], 1, 1, 1)
        transition_score_norm = v_dom / v_eff
        return th.where(
            score_norm <= transition_score_norm,
            v_eff,
            v_dom / score_norm.clamp_min(eps),
        )

    @staticmethod
    def compute_divergence(
        score: th.Tensor,
        previous_score: th.Tensor,
        stepsize: th.Tensor,
    ) -> Tuple[th.Tensor, th.Tensor]:
        """Estimate one state-dependent Fisher energy increment.

        This is ``v_infinity**2 * mean(||score - previous_score||**2)``, where
        ``v_infinity`` is the maximum lower-level step size over the particles.
        """
        if score.shape != previous_score.shape:
            raise ValueError(
                "score and previous_score must have identical shapes, got "
                f"{tuple(score.shape)} and {tuple(previous_score.shape)}"
            )
        if score.ndim < 2:
            raise ValueError("scores must include batch and feature dimensions.")
        if stepsize.numel() == 0:
            raise ValueError("stepsize must not be empty.")

        stepsize = stepsize.to(device=score.device, dtype=score.dtype)
        v_infinity = stepsize.max()
        squared_score_difference = (score - previous_score).reshape(
            score.shape[0],
            -1,
        )
        mean_squared_score_difference = (
            squared_score_difference.square().sum(dim=1).mean()
        )
        energy_increment = v_infinity.square() * mean_squared_score_difference
        return energy_increment, v_infinity

    def compute_guidance(
        self,
        noisy_images: th.Tensor,
        schedule_values: th.Tensor,
        unconditional: bool = False,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        """Compute likelihood guidance, prior score, likelihood, and clean estimate."""
        schedule_values = schedule_values.to(
            device=noisy_images.device,
            dtype=noisy_images.dtype,
        )
        prior_score = self.compute_prior_score(noisy_images, schedule_values)

        if unconditional:
            likelihood_guidance = th.zeros_like(noisy_images)
            log_likelihood = noisy_images.new_zeros((noisy_images.shape[0],))
            clean_estimate = th.zeros_like(noisy_images)
            return (
                likelihood_guidance,
                prior_score.detach(),
                log_likelihood,
                clean_estimate,
            )

        clean_estimate = self.tweedie_denoise(
            noisy_images,
            schedule_values,
            prior_score=prior_score,
        )
        positive_clean_estimate = th.clamp(clean_estimate + 1.0, min=1e-15)
        log_likelihood = self.log_likelihood(positive_clean_estimate)
        if not isinstance(log_likelihood, th.Tensor):
            raise TypeError("log_likelihood must return a torch.Tensor.")
        expected_shape = (noisy_images.shape[0],)
        if log_likelihood.shape != expected_shape:
            raise ValueError(
                "log_likelihood must return one value per sample with shape "
                f"{expected_shape}, got {tuple(log_likelihood.shape)}"
            )

        likelihood_guidance = th.autograd.grad(
            outputs=log_likelihood,
            inputs=noisy_images,
            grad_outputs=th.ones_like(log_likelihood),
        )[0]
        clean_estimate = positive_clean_estimate.detach() - 1.0
        return likelihood_guidance, prior_score, log_likelihood, clean_estimate

    def conditional_corrector_step(
        self,
        images: th.Tensor,
        schedule_values: th.Tensor,
        num_steps: int,
        guidance_strength: float,
        v_eff: Scalar,
        v_dom: Scalar,
        *,
        eps: float = 1e-12,
    ) -> th.Tensor:
        """Apply conditional Euler-Maruyama corrector steps."""
        if num_steps < 0:
            raise ValueError("num_steps must be non-negative.")

        for _ in range(num_steps):
            images = images.detach().requires_grad_(True)
            use_likelihood = (
                float(guidance_strength) > 0.0
                and self.log_likelihood is not zero_log_likelihood
            )
            likelihood_guidance, prior_score, _, _ = self.compute_guidance(
                images,
                schedule_values,
                unconditional=not use_likelihood,
            )
            conditional_score = (
                prior_score + float(guidance_strength) * likelihood_guidance
            )
            stepsize = self.compute_stepsize(
                conditional_score,
                v_eff=v_eff,
                v_dom=v_dom,
                eps=eps,
            )
            noise = th.randn_like(images)
            images = (
                images
                + stepsize * conditional_score
                + th.sqrt(2.0 * stepsize) * noise
            ).detach()

        return images.requires_grad_(True)

    def compute_conditional_score(
        self,
        images: th.Tensor,
        schedule_values: th.Tensor,
        guidance_strength: float,
        return_mean_log_likelihood: bool = False,
    ) -> Tuple[th.Tensor, Union[float, th.Tensor]]:
        """Return the conditional score and its likelihood diagnostic."""
        use_likelihood = (
            float(guidance_strength) > 0.0
            and self.log_likelihood is not zero_log_likelihood
        )
        likelihood_guidance, prior_score, log_likelihood, _ = self.compute_guidance(
            images,
            schedule_values,
            unconditional=not use_likelihood,
        )
        conditional_score = (
            prior_score + float(guidance_strength) * likelihood_guidance
        )

        if return_mean_log_likelihood:
            mean_log_likelihood = float(log_likelihood.detach().mean().item())
            return conditional_score, mean_log_likelihood
        return conditional_score, log_likelihood
