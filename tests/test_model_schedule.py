import unittest

import numpy as np
import torch as th

from model_schedule import (
    estimate_model_schedule,
    install_diffusion_schedule,
    schedule_from_weighted_score_energy,
    vp_betas_from_omega,
)


class TimeOnlyEpsilonModel(th.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = th.nn.Parameter(th.zeros(()))

    def forward(self, images, times):
        return (
            th.ones_like(images) * times.view(-1, 1, 1, 1)
            + self.anchor * 0.0
        )


class DummyDiffusion:
    def __init__(self):
        self.omega = np.asarray([0.1, 0.3, 0.7], dtype=np.float64)
        self.betas = np.asarray([0.1, 0.2, 0.3], dtype=np.float64)
        self.alphas_cumprod = np.asarray([0.9, 0.7, 0.4], dtype=np.float64)


class InstallableDiffusion:
    def __init__(self):
        self.num_timesteps = 1000
        self.lambda_increments = th.ones(1000)
        self.energy_increments = th.ones(1000)
        self.D = th.ones(1000)
        self.D2 = th.ones(1000)
        self.n_time = th.ones(1000, dtype=th.int64)

    def update_alpha(self):
        self.alphas_cumprod = np.cumprod(1.0 - self.betas)


class ModelScheduleTests(unittest.TestCase):
    def test_install_schedule_updates_loaded_timestep_count(self):
        diffusion = InstallableDiffusion()
        omega = np.asarray([0.1, 0.3, 0.7], dtype=np.float64)
        betas = np.asarray([0.05, 0.1, 0.2], dtype=np.float64)

        install_diffusion_schedule(diffusion, omega=omega, betas=betas)

        self.assertEqual(diffusion.num_timesteps, 3)
        np.testing.assert_array_equal(diffusion.omega, omega)
        np.testing.assert_array_equal(diffusion.betas, betas)
        np.testing.assert_allclose(
            diffusion.alphas_cumprod,
            np.cumprod(1.0 - betas),
        )
        for name in ("lambda_increments", "energy_increments", "D", "D2"):
            self.assertEqual(getattr(diffusion, name).shape, (3,))
        self.assertEqual(diffusion.n_time.shape, (3,))
        self.assertEqual(diffusion.n_time[0].item(), 1)

    def test_install_schedule_rejects_mismatched_lengths(self):
        with self.assertRaisesRegex(ValueError, "equal lengths"):
            install_diffusion_schedule(
                InstallableDiffusion(),
                omega=np.asarray([0.1, 0.3, 0.7]),
                betas=np.asarray([0.1, 0.2]),
            )

    def test_exact_vp_betas_reproduce_integrated_noise_time(self):
        omega = np.asarray([0.1, 0.3, 1.2], dtype=np.float64)
        betas = vp_betas_from_omega(omega)

        alpha_bar = np.cumprod(1.0 - betas)
        np.testing.assert_allclose(alpha_bar, np.exp(-omega))
        self.assertTrue(np.all((betas > 0.0) & (betas < 1.0)))

    def test_equal_energy_preserves_existing_schedule(self):
        omega = np.asarray([0.1, 0.2, 0.4, 0.8], dtype=np.float64)
        estimate = schedule_from_weighted_score_energy(
            omega=omega,
            energy_increments=np.ones(3),
        )

        np.testing.assert_allclose(estimate.optimized_omega, omega)
        np.testing.assert_allclose(
            estimate.normalized_cumulative_length,
            np.linspace(0.0, 1.0, omega.size),
        )

    def test_nonuniform_energy_moves_nodes_toward_large_increment(self):
        omega = np.asarray([0.1, 0.2, 0.4, 0.8], dtype=np.float64)
        estimate = schedule_from_weighted_score_energy(
            omega=omega,
            energy_increments=np.asarray([1.0, 9.0, 1.0]),
        )

        self.assertEqual(estimate.optimized_omega[0], omega[0])
        self.assertEqual(estimate.optimized_omega[-1], omega[-1])
        self.assertTrue(np.all(np.diff(estimate.optimized_omega) > 0.0))
        self.assertGreater(estimate.optimized_omega[1], omega[1])
        self.assertLess(estimate.optimized_omega[2], omega[2])

    def test_estimator_matches_trainer_weighted_score_formula(self):
        diffusion = DummyDiffusion()
        model = TimeOnlyEpsilonModel()
        samples = th.zeros(3, 1, 2, 2)
        estimate = estimate_model_schedule(
            model=model,
            diffusion=diffusion,
            clean_samples=samples,
            batch_size=2,
            device=th.device("cpu"),
            seed=4,
        )

        sigma = np.sqrt(1.0 - diffusion.alphas_cumprod)
        expected = []
        for index in range(1, diffusion.omega.size):
            score_previous = -diffusion.omega[index - 1] / sigma[index - 1]
            score_current = -diffusion.omega[index] / sigma[index]
            expected.append(
                sigma[index - 1] ** 2
                * 4
                * (score_previous - score_current) ** 2
            )
        np.testing.assert_allclose(
            estimate.energy_increments,
            expected,
            rtol=1e-6,
        )
        self.assertEqual(estimate.num_samples, samples.shape[0])
        np.testing.assert_array_equal(
            estimate.original_betas,
            diffusion.betas,
        )


if __name__ == "__main__":
    unittest.main()
