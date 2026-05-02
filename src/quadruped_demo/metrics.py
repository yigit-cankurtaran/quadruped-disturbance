from __future__ import annotations

import numpy as np


def body_height(qpos: np.ndarray) -> float:
    return float(qpos[2])


def forward_distance(qpos: np.ndarray) -> float:
    return float(qpos[0])


def is_upright(qpos: np.ndarray, min_height: float = 0.18) -> bool:
    # TODO: improve the detection on this
    return body_height(qpos) >= min_height
