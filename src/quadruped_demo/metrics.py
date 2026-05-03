from __future__ import annotations

import math

import numpy as np


def body_height(qpos: np.ndarray) -> float:
    return float(qpos[2])


def forward_distance(qpos: np.ndarray) -> float:
    return float(qpos[0])


def base_roll_pitch(qpos: np.ndarray) -> tuple[float, float]:
    """Return floating-base roll and pitch in radians from MuJoCo qpos."""

    # mujoco quaternion order is wxyz, NOT xyzw
    qw, qx, qy, qz = np.asarray(qpos[3:7], dtype=np.float64)

    # quaternion length
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if norm == 0.0:
        raise ValueError("Base orientation quaternion has zero norm.")

    # normalize quaternion to make it more robust
    qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm

    # roll = rotation around x-axis
    roll = math.atan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))

    # pitch = rotation around y-axis
    pitch_arg = 2.0 * (qw * qy - qz * qx)
    pitch = math.asin(float(np.clip(pitch_arg, -1.0, 1.0)))
    return roll, pitch


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def base_yaw(qpos: np.ndarray) -> float:
    """Return floating-base yaw in radians from MuJoCo qpos."""

    qw, qx, qy, qz = np.asarray(qpos[3:7], dtype=np.float64)
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if norm == 0.0:
        raise ValueError("Base orientation quaternion has zero norm.")

    qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm
    return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def is_upright(
    qpos: np.ndarray,
    min_height: float = 0.18,
    max_tilt_rad: float = math.radians(55.0),
) -> bool:
    if body_height(qpos) < min_height:
        return False
    roll, pitch = base_roll_pitch(qpos)
    return abs(roll) <= max_tilt_rad and abs(pitch) <= max_tilt_rad
