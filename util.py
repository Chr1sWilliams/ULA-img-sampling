from pathlib import Path

import numpy as np


def load_schedule(path: str, target_length: int) -> np.ndarray:
    """Load, validate, and optionally resample a one-dimensional schedule."""
    schedule_path = Path(path)
    if not schedule_path.is_file():
        raise FileNotFoundError(f"Schedule file does not exist: {schedule_path}")
    if target_length < 2:
        raise ValueError("target_length must be at least 2.")

    schedule = np.asarray(np.load(schedule_path), dtype=np.float64).reshape(-1)
    if schedule.size < 2:
        raise ValueError("The schedule must contain at least two values.")
    if not np.all(np.isfinite(schedule)):
        raise ValueError("The schedule contains non-finite values.")
    if np.any(np.diff(schedule) < 0):
        raise ValueError("The schedule must be monotonically non-decreasing.")

    if schedule.size != target_length:
        schedule = np.interp(
            np.linspace(0.0, 1.0, target_length),
            np.linspace(0.0, 1.0, schedule.size),
            schedule,
        )
    return schedule
