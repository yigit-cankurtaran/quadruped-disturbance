from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import mujoco
import numpy as np

from quadruped_demo.paths import require_model

Controller = Callable[[float, dict[str, np.ndarray]], np.ndarray]


@dataclass  # using dataclass because it just needs to hold data
class PushConfig:
    interval_s: float = 2.0
    duration_s: float = 0.08
    force_n: float = 35.0


class Go1Env:
    """small custom MuJoCo wrapper for the Unitree Go1 model."""

    def __init__(self, model_path: str | None = None) -> None:
        xml_path = model_path or str(require_model())
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        # trunk in mujoco=central body, anchor point for limbs and joints
        self.trunk_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
        self.reset()

    @property
    def dt(self) -> float:
        # exposing timestep for convenience
        return float(self.model.opt.timestep)

    def reset(self) -> dict[str, np.ndarray]:
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)

        # recomputing data from the new state before stepping
        mujoco.mj_forward(self.model, self.data)
        return self.observation()

    def observation(self) -> dict[str, np.ndarray]:
        return {
            "qpos": self.data.qpos.copy(),
            "qvel": self.data.qvel.copy(),
            "ctrl": self.data.ctrl.copy(),
            "time": np.array([self.data.time], dtype=np.float64),
        }

    def step(self, ctrl: np.ndarray) -> dict[str, np.ndarray]:
        ctrl_min = self.model.actuator_ctrlrange[:, 0]
        ctrl_max = self.model.actuator_ctrlrange[:, 1]
        self.data.ctrl[:] = np.clip(ctrl, ctrl_min, ctrl_max)
        mujoco.mj_step(self.model, self.data)
        return self.observation()

    def apply_push(self, force_xyz: np.ndarray) -> None:
        # xfrc_applied stores external wrench values applied to bodies.
        # [:3] bc i only want linear vals, not torque
        self.data.xfrc_applied[self.trunk_id, :3] = force_xyz

    def clear_push(self) -> None:
        self.data.xfrc_applied[self.trunk_id, :] = 0.0


def run_controller(
    controller: Controller,
    duration_s: float,
    push: PushConfig | None = None,
    render: bool = False,
) -> Go1Env:
    env = Go1Env()
    viewer = None
    if render:
        import mujoco.viewer

        viewer = mujoco.viewer.launch_passive(env.model, env.data)

    start = time.time()
    while env.data.time < duration_s:
        obs = env.observation()
        ctrl = controller(float(env.data.time), obs)

        if push is not None and (env.data.time % push.interval_s) < push.duration_s:
            direction = -1.0 if int(env.data.time / push.interval_s) % 2 else 1.0
            env.apply_push(np.array([0.0, direction * push.force_n, 0.0], dtype=np.float64))
        else:
            env.clear_push()

        env.step(ctrl)
        if viewer is not None:
            viewer.sync()
            elapsed = time.time() - start
            if env.data.time > elapsed:
                time.sleep(env.data.time - elapsed)

    if viewer is not None:
        viewer.close()
    return env


def launch_viewer() -> None:
    import mujoco.viewer

    env = Go1Env()
    mujoco.viewer.launch(env.model, env.data)
