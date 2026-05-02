from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from quadruped_demo.gait import HOME_POSE, TrotGait


class StandingController:
    """hold the Menagerie home keyframe using the model's built-in position actuators."""

    def __init__(self, target: np.ndarray = HOME_POSE) -> None:
        self.target = np.asarray(target, dtype=np.float64)

    def __call__(self, time_s: float, obs: dict[str, np.ndarray]) -> np.ndarray:
        del time_s, obs
        return self.target.copy()


@dataclass
class TrotController:
    gait: TrotGait = TrotGait()
    base_pose: np.ndarray = field(default_factory=lambda: HOME_POSE.copy())

    def __call__(self, time_s: float, obs: dict[str, np.ndarray]) -> np.ndarray:
        del obs
        return self.gait.target(time_s, self.base_pose)
