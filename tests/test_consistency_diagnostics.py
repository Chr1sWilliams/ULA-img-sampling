import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch as th

from consistency_diagnostics import (
    ConsistencyPath,
    estimate_consistency_path,
    forward_diffusion_sample,
    load_selected_threshold,
    save_consistency_path,
    select_consistency_threshold,
    tweedie_reconstruction,
)


class ConsistencyDiagnosticsTests(unittest.TestCase):
    def test_selects_first_crossing_from_high_noise(self):
        schedule = np.asarray([0.1, 0.2, 0.3, 0.4])
        eta = np.asarray([0.01, 0.04, 0.09, 0.3])
        index, value = select_consistency_threshold(
            schedule,
            eta,
            tolerance=0.1,
        )
        self.assertEqual(index, 2)
        self.assertAlmostEqual(value, 0.3)

    def test_returns_none_without_crossing(self):
        index, value = select_consistency_threshold(
            np.asarray([0.1, 0.2]),
            np.asarray([0.3, 0.4]),
            tolerance=0.2,
        )
        self.assertIsNone(index)
        self.assertIsNone(value)

    def test_forward_kernel_and_tweedie_formula(self):
        clean = th.ones(2, 1, 3, 3)
        noise = th.zeros_like(clean)
        schedule_value = 0.25
        noisy = forward_diffusion_sample(clean, schedule_value, noise)
        self.assertTrue(th.allclose(noisy, clean * 0.75))

        standard_normal_score = -noisy
        reconstruction = tweedie_reconstruction(
            noisy,
            schedule_value,
            standard_normal_score,
        )
        self.assertTrue(th.allclose(reconstruction, noisy * 0.75))

    def test_saves_path_and_selected_threshold(self):
        schedule = np.asarray([0.1, 0.2])
        result = ConsistencyPath(
            schedule=schedule,
            eta=np.asarray([0.05, 0.2]),
            eta_squared=np.asarray([0.0025, 0.04]),
            numerator=np.asarray([1.0, 4.0]),
            denominator=np.asarray([400.0, 100.0]),
            reconstruction_mse=np.asarray([0.01, 0.03]),
            clean_score_rms=np.asarray([1.0, 1.0]),
            reconstructed_score_rms=np.asarray([1.0, 0.9]),
            num_samples=np.asarray([4, 4]),
            tolerance=0.1,
            selected_index=0,
            selected_schedule_value=0.1,
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            save_consistency_path(result, output)
            self.assertTrue((output / "consistency_path.csv").is_file())
            self.assertTrue((output / "consistency_path.npz").is_file())
            self.assertTrue((output / "consistency_path.png").is_file())
            self.assertTrue((output / "summary.json").is_file())
            threshold = float(np.load(output / "guidance_threshold.npy"))
            self.assertAlmostEqual(threshold, 0.1)

    def test_estimates_complete_path(self):
        schedule = np.asarray([0.1, 0.2])

        def clean_batches(_level_index):
            yield th.ones(3, 1, 2, 2)

        def prior_score(noisy_images, _schedule_values):
            return -noisy_images

        result = estimate_consistency_path(
            schedule=schedule,
            clean_batch_factory=clean_batches,
            prior_score=prior_score,
            likelihood_score=lambda images: images,
            tolerance=10.0,
            device=th.device("cpu"),
            seed=7,
        )
        self.assertEqual(result.eta.shape, schedule.shape)
        self.assertTrue(np.isfinite(result.eta).all())
        self.assertEqual(result.num_samples.tolist(), [3, 3])
        self.assertEqual(result.selected_index, 1)
        self.assertAlmostEqual(result.selected_schedule_value, 0.2)

    def test_loads_threshold_from_result_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "summary.json").write_text(
                json.dumps({"selected_schedule_value": 0.275})
            )
            self.assertAlmostEqual(load_selected_threshold(directory), 0.275)


if __name__ == "__main__":
    unittest.main()
