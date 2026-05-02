from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

N_ACTUATORS = 12

# Menagerie Go1 actuator order:
# FR hip/thigh/calf, FL hip/thigh/calf, RR hip/thigh/calf, RL hip/thigh/calf.
HOME_POSE = np.array(
    # 12 positions for 12 actuators
    [0.0, 0.9, -1.8, 0.0, 0.9, -1.8, 0.0, 0.9, -1.8, 0.0, 0.9, -1.8],
    dtype=np.float64,
)


@dataclass(frozen=True)
class TrotGait:
    frequency_hz: float = 1.6  # cadence, step speed
    thigh_amplitude: float = 0.22  # stride length
    calf_amplitude: float = 0.32  # clearance
    hip_sway_amplitude: float = 0.035  # side to side motion

    def target(self, time_s: float, base_pose: np.ndarray = HOME_POSE) -> np.ndarray:
        """open-loop (no correction) diagonal-pair
        (FR and RL, FL and RR) trot target for position actuators."""

        # i plan to add reactivity with RL instead of control.
        ctrl = np.asarray(base_pose, dtype=np.float64).copy()
        phase = 2.0 * math.pi * self.frequency_hz * time_s
        diagonal_phase = {
            "FR": 0.0,  # front right
            "FL": math.pi,  # front left
            "RR": math.pi,  # rear right
            "RL": 0.0,  # rear left
        }
        leg_offsets = {"FR": 0, "FL": 3, "RR": 6, "RL": 9}

        for leg, offset in leg_offsets.items():
            leg_phase = phase + diagonal_phase[leg]
            swing = math.sin(leg_phase)
            lift = max(0.0, swing)
            ctrl[offset] += self.hip_sway_amplitude * math.sin(leg_phase + math.pi / 2.0)
            ctrl[offset + 1] += self.thigh_amplitude * swing
            ctrl[offset + 2] += self.calf_amplitude * lift - 0.08 * min(0.0, swing)

        return ctrl
