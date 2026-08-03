import tempfile
import unittest
from pathlib import Path

import numpy as np

from util import load_schedule


class LoadScheduleTests(unittest.TestCase):
    def test_interpolates_preloaded_schedule_to_more_points(self):
        original = np.asarray([0.1, 0.3, 0.8], dtype=np.float64)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "schedule.npy"
            np.save(path, original)
            loaded = load_schedule(str(path), target_length=7)

        self.assertEqual(loaded.shape, (7,))
        self.assertEqual(loaded[0], original[0])
        self.assertEqual(loaded[-1], original[-1])
        self.assertTrue(np.all(np.diff(loaded) >= 0.0))

    def test_interpolates_preloaded_schedule_to_fewer_points(self):
        original = np.asarray(
            [0.1, 0.2, 0.35, 0.55, 0.8],
            dtype=np.float64,
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "schedule.npy"
            np.save(path, original)
            loaded = load_schedule(str(path), target_length=3)

        self.assertEqual(loaded.shape, (3,))
        self.assertEqual(loaded[0], original[0])
        self.assertEqual(loaded[-1], original[-1])
        self.assertTrue(np.all(np.diff(loaded) >= 0.0))


if __name__ == "__main__":
    unittest.main()
