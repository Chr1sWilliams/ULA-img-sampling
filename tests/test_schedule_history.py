import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch as th

from schedule_diagnostics import (
    build_round_diagnostics,
    save_round_artifacts,
    save_run_history,
)


def make_diagnostics(schedule, energy, clip_multiple=np.inf):
    schedule = np.asarray(schedule, dtype=np.float64)
    node_values = np.linspace(0.1, 0.9, schedule.size)
    return build_round_diagnostics(
        schedule=schedule,
        energy_increments=np.asarray(energy, dtype=np.float64),
        v_infinity=np.ones(schedule.size - 1),
        diffusion_time=node_values,
        guidance_weight=node_values[::-1],
        sigma=node_values,
        v_eff=node_values,
        v_dom=node_values,
        length_increment_clip_multiple=clip_multiple,
    )


class ScheduleHistoryTests(unittest.TestCase):
    def test_default_infinite_clip_leaves_increments_unchanged(self):
        diagnostics = make_diagnostics(
            [0.05, 0.3, 0.7, 0.95],
            [1.0, 4.0, 100.0],
        )

        np.testing.assert_array_equal(
            diagnostics.energy_increments,
            diagnostics.raw_energy_increments,
        )
        np.testing.assert_array_equal(
            diagnostics.length_increments,
            diagnostics.raw_length_increments,
        )
        self.assertEqual(diagnostics.num_length_increments_clipped, 0)
        self.assertTrue(np.isinf(diagnostics.length_increment_clip_threshold))

    def test_infinite_clip_handles_zero_length_path(self):
        diagnostics = make_diagnostics(
            [0.05, 0.3, 0.7, 0.95],
            [0.0, 0.0, 0.0],
        )

        np.testing.assert_array_equal(
            diagnostics.length_increments,
            np.zeros(3),
        )
        np.testing.assert_array_equal(
            diagnostics.schedule_after,
            diagnostics.schedule_before,
        )
        self.assertFalse(np.isnan(diagnostics.total_length))

    def test_finite_clip_caps_length_at_multiple_of_raw_mean(self):
        diagnostics = make_diagnostics(
            [0.05, 0.3, 0.7, 0.95],
            [1.0, 1.0, 100.0],
            clip_multiple=1.0,
        )

        np.testing.assert_allclose(
            diagnostics.raw_length_increments,
            [1.0, 1.0, 10.0],
        )
        np.testing.assert_allclose(
            diagnostics.length_increments,
            [1.0, 1.0, 4.0],
        )
        np.testing.assert_allclose(
            diagnostics.energy_increments,
            [1.0, 1.0, 16.0],
        )
        self.assertEqual(diagnostics.raw_total_energy, 102.0)
        self.assertEqual(diagnostics.total_energy, 18.0)
        self.assertEqual(diagnostics.raw_total_length, 12.0)
        self.assertEqual(diagnostics.total_length, 6.0)
        self.assertEqual(diagnostics.length_increment_clip_threshold, 4.0)
        self.assertEqual(diagnostics.num_length_increments_clipped, 1)

    def test_saves_round_samples_and_progression(self):
        first = make_diagnostics(
            [0.05, 0.3, 0.7, 0.95],
            [1.0, 4.0, 9.0],
        )
        second = make_diagnostics(
            first.schedule_after,
            [2.0, 3.0, 5.0],
        )
        samples = th.arange(4 * 4 * 4, dtype=th.float32).reshape(
            4,
            1,
            4,
            4,
        )

        with tempfile.TemporaryDirectory() as directory:
            run_directory = Path(directory)
            round_directory = save_round_artifacts(
                run_directory=run_directory,
                round_index=7,
                diagnostics=first,
                samples=samples,
                metadata={"dataset": "test"},
            )
            self.assertTrue((round_directory / "samples.pt").is_file())
            self.assertTrue((round_directory / "sample_mean.npy").is_file())
            self.assertTrue((round_directory / "sample_std.npy").is_file())
            self.assertTrue(
                (round_directory / "sample_statistics.json").is_file()
            )
            self.assertTrue(
                (round_directory / "round_diagnostics.png").is_file()
            )
            self.assertTrue(
                (round_directory / "sample_preview.png").is_file()
            )
            self.assertTrue(
                (round_directory / "raw_length_increments.npy").is_file()
            )

            save_run_history(
                run_directory=run_directory,
                schedule_history=[
                    first.schedule_before,
                    first.schedule_after,
                    second.schedule_after,
                ],
                diagnostics_history=[first, second],
                round_indices=[7, 8],
            )

            self.assertTrue((run_directory / "run_history.npz").is_file())
            self.assertTrue(
                (run_directory / "schedule_progression.png").is_file()
            )
            self.assertTrue(
                (run_directory / "energy_length_progression.png").is_file()
            )
            with (run_directory / "round_progress.csv").open() as csv_file:
                rows = list(csv.DictReader(csv_file))
            self.assertEqual([row["round_index"] for row in rows], ["7", "8"])

            history = np.load(run_directory / "run_history.npz")
            np.testing.assert_array_equal(history["round_index"], [7, 8])
            self.assertEqual(history["schedule_history"].shape, (3, 4))
            self.assertIn("raw_length_increment_history", history.files)
            self.assertIn(
                "num_length_increments_clipped_history",
                history.files,
            )


if __name__ == "__main__":
    unittest.main()
