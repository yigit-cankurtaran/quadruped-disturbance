from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from quadruped_demo.env import Go1Env
from quadruped_demo.gait import N_ACTUATORS, TrotGait
from quadruped_demo.metrics import base_roll_pitch, base_yaw, wrap_angle

ROOT_QPOS_SIZE = 7  # x,y,z,qw,qx,qy,qz
ROOT_QVEL_SIZE = 6  # linear x,y,z and angular x,y,z

EXPECTED_ACTUATOR_NAMES = (
    "FR_hip",
    "FR_thigh",
    "FR_calf",
    "FL_hip",
    "FL_thigh",
    "FL_calf",
    "RR_hip",
    "RR_thigh",
    "RR_calf",
    "RL_hip",
    "RL_thigh",
    "RL_calf",
)


@dataclass(frozen=True)
class RandomPushConfig:
    enabled: bool = False
    min_interval_s: float = 1.0
    max_interval_s: float = 3.0
    min_duration_s: float = 0.04
    max_duration_s: float = 0.12
    min_force_n: float = 10.0
    max_force_n: float = 45.0


@dataclass(frozen=True)
class ResetRandomizationConfig:
    enabled: bool = False
    base_xy_range_m: float = 0.02
    base_yaw_range_rad: float = 0.05
    root_velocity_range: float = 0.05
    joint_position_range_rad: float = 0.03
    joint_velocity_range: float = 0.2
    randomize_gait_phase: bool = True


@dataclass(frozen=True)
class RewardConfig:
    velocity_tracking_weight: float = 1.0
    velocity_tracking_sigma: float = 0.35
    straight_line_weight: float = 0.3
    straight_line_sigma: float = 0.25
    lateral_velocity_weight: float = 0.1
    heading_weight: float = 0.25
    heading_sigma: float = 0.35
    yaw_rate_weight: float = 0.05
    alive_weight: float = 0.5
    height_weight: float = 0.4
    height_sigma: float = 0.05
    roll_pitch_weight: float = 0.25
    action_weight: float = 0.02
    action_rate_weight: float = 0.01
    energy_weight: float = 0.0005


@dataclass(frozen=True)
class Go1RLEnvConfig:
    model_path: str | None = None
    gait: TrotGait = field(default_factory=TrotGait)
    action_scale: float = 0.25  # normalize and multiply RL action before adding to gait
    decimation: int = 5  # 1 RL step = 5 mujoco physics steps
    max_episode_time_s: float = 10.0
    command_velocity_x: float = 0.6
    command_velocity_range: tuple[float, float] = (-1.0, 2.0)
    randomize_command_velocity: bool = False
    target_height: float = 0.27
    min_base_height: float = 0.16
    max_abs_roll_pitch_rad: float = math.radians(55.0)
    reward: RewardConfig = field(default_factory=RewardConfig)
    random_pushes: RandomPushConfig = field(default_factory=RandomPushConfig)
    reset_randomization: ResetRandomizationConfig = field(
        default_factory=ResetRandomizationConfig
    )


class Go1TrotRLEnv(gym.Env[np.ndarray, np.ndarray]):
    """Gymnasium task env for residual control over the scripted Go1 trot."""

    metadata = {"render_modes": []}

    def __init__(self, config: Go1RLEnvConfig | None = None) -> None:
        self.config = config or Go1RLEnvConfig()
        self._validate_config()

        self.physics = Go1Env(model_path=self.config.model_path)
        self.model = self.physics.model
        self.data = self.physics.data

        self._validate_model_layout()
        self._joint_qpos_adrs, self._joint_qvel_adrs = self._actuated_joint_addresses()
        self._ctrl_min = self.model.actuator_ctrlrange[:, 0].copy()
        self._ctrl_max = self.model.actuator_ctrlrange[:, 1].copy()
        self._joint_pos_low = self.model.jnt_range[self.model.actuator_trnid[:, 0], 0].copy()
        self._joint_pos_high = self.model.jnt_range[self.model.actuator_trnid[:, 0], 1].copy()

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(N_ACTUATORS,),
            dtype=np.float32,
        )
        self.observation_space = self._make_observation_space()

        self._previous_action = np.zeros(N_ACTUATORS, dtype=np.float64)
        self._episode_start_time = 0.0
        self._episode_start_y = float(self.data.qpos[1])
        self._command_velocity_x = self.config.command_velocity_x
        self._push_force = np.zeros(3, dtype=np.float64)
        self._next_push_time = math.inf
        self._push_end_time = -math.inf

    @property
    def dt(self) -> float:
        return self.physics.dt * self.config.decimation

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.physics.reset()
        self._previous_action.fill(0.0)
        self._command_velocity_x = self._command_velocity_from_options(options)
        self._apply_reset_randomization()
        self._episode_start_time = float(self.data.time)
        self._episode_start_y = float(self.data.qpos[1])
        self._push_force.fill(0.0)
        self._push_end_time = -math.inf
        self.physics.clear_push()
        self._schedule_next_push(float(self.data.time))

        obs = self._observation()
        return obs, self._info({})

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        normalized_action = self._normalized_action(action)
        previous_action = self._previous_action.copy()
        action_rate = normalized_action - previous_action

        if self._is_terminated():
            reward, reward_terms = self._reward(normalized_action, action_rate, terminated=True)
            self._previous_action = normalized_action.copy()
            obs = self._observation()
            return obs, reward, True, False, self._info(reward_terms)

        for _ in range(self.config.decimation):
            self._update_push(float(self.data.time))
            base_target = self.config.gait.target(float(self.data.time))
            ctrl = np.clip(
                base_target + self.config.action_scale * normalized_action,
                self._ctrl_min,
                self._ctrl_max,
            )
            self.physics.step(ctrl)

        self._update_push(float(self.data.time))
        terminated = self._is_terminated()
        truncated = self._is_truncated()
        reward, reward_terms = self._reward(normalized_action, action_rate, terminated)
        self._previous_action = normalized_action.copy()

        obs = self._observation()
        return obs, reward, terminated, truncated, self._info(reward_terms)

    def close(self) -> None:
        self.physics.clear_push()

    def _validate_config(self) -> None:
        if self.config.decimation < 1:
            raise ValueError("decimation must be >= 1")
        if self.config.action_scale < 0.0:
            raise ValueError("action_scale must be >= 0")
        if self.config.max_episode_time_s <= 0.0:
            raise ValueError("max_episode_time_s must be > 0")
        if self.config.target_height <= 0.0:
            raise ValueError("target_height must be > 0")
        if self.config.min_base_height <= 0.0:
            raise ValueError("min_base_height must be > 0")
        if self.config.max_abs_roll_pitch_rad <= 0.0:
            raise ValueError("max_abs_roll_pitch_rad must be > 0")

        command_low, command_high = self.config.command_velocity_range
        if command_low > command_high:
            raise ValueError("command_velocity_range must be ordered low <= high")

        self._validate_random_push_config(self.config.random_pushes)
        self._validate_reset_randomization_config(self.config.reset_randomization)

    @staticmethod
    def _validate_random_push_config(config: RandomPushConfig) -> None:
        ranges = {
            "interval": (config.min_interval_s, config.max_interval_s),
            "duration": (config.min_duration_s, config.max_duration_s),
            "force": (config.min_force_n, config.max_force_n),
        }
        for name, (low, high) in ranges.items():
            if low < 0.0 or high < 0.0:
                raise ValueError(f"random push {name} range must be non-negative")
            if low > high:
                raise ValueError(f"random push {name} range must be ordered low <= high")

    @staticmethod
    def _validate_reset_randomization_config(config: ResetRandomizationConfig) -> None:
        values = {
            "base_xy_range_m": config.base_xy_range_m,
            "base_yaw_range_rad": config.base_yaw_range_rad,
            "root_velocity_range": config.root_velocity_range,
            "joint_position_range_rad": config.joint_position_range_rad,
            "joint_velocity_range": config.joint_velocity_range,
        }
        for name, value in values.items():
            if value < 0.0:
                raise ValueError(f"{name} must be >= 0")

    def _validate_model_layout(self) -> None:
        if self.model.nu != N_ACTUATORS:
            raise ValueError(f"Expected {N_ACTUATORS} actuators, got {self.model.nu}.")
        if self.model.nq < ROOT_QPOS_SIZE + N_ACTUATORS:
            raise ValueError(
                f"Expected freejoint qpos plus {N_ACTUATORS} joints, got nq={self.model.nq}."
            )
        if self.model.nv < ROOT_QVEL_SIZE + N_ACTUATORS:
            raise ValueError(
                f"Expected freejoint qvel plus {N_ACTUATORS} dofs, got nv={self.model.nv}."
            )

        actuator_names = tuple(
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            for i in range(self.model.nu)
        )
        if actuator_names != EXPECTED_ACTUATOR_NAMES:
            raise ValueError(f"Unexpected Go1 actuator order: {actuator_names}")

    def _actuated_joint_addresses(self) -> tuple[np.ndarray, np.ndarray]:
        joint_ids = self.model.actuator_trnid[:, 0]
        qpos_adrs = self.model.jnt_qposadr[joint_ids]
        qvel_adrs = self.model.jnt_dofadr[joint_ids]
        return qpos_adrs.astype(np.int64), qvel_adrs.astype(np.int64)

    def _make_observation_space(self) -> spaces.Box:
        command_low, command_high = self.config.command_velocity_range
        low = np.concatenate(
            [
                np.array([0.0], dtype=np.float32),
                np.array([-np.inf], dtype=np.float32),
                np.full(2, -1.0, dtype=np.float32),
                np.full(3, -1.0, dtype=np.float32),
                np.full(6, -np.inf, dtype=np.float32),
                self._joint_pos_low.astype(np.float32),
                np.full(N_ACTUATORS, -np.inf, dtype=np.float32),
                np.full(N_ACTUATORS, -1.0, dtype=np.float32),
                np.full(2, -1.0, dtype=np.float32),
                np.array([command_low], dtype=np.float32),
            ]
        )
        high = np.concatenate(
            [
                np.array([2.0], dtype=np.float32),
                np.array([np.inf], dtype=np.float32),
                np.full(2, 1.0, dtype=np.float32),
                np.full(3, 1.0, dtype=np.float32),
                np.full(6, np.inf, dtype=np.float32),
                self._joint_pos_high.astype(np.float32),
                np.full(N_ACTUATORS, np.inf, dtype=np.float32),
                np.full(N_ACTUATORS, 1.0, dtype=np.float32),
                np.full(2, 1.0, dtype=np.float32),
                np.array([command_high], dtype=np.float32),
            ]
        )
        return spaces.Box(low=low, high=high, dtype=np.float32)

    def _observation(self) -> np.ndarray:
        qpos = self.data.qpos
        qvel = self.data.qvel
        phase = self._gait_phase(float(self.data.time))

        obs = np.concatenate(
            [
                np.array([qpos[2]], dtype=np.float64),
                np.array([self._lateral_displacement()], dtype=np.float64),
                self._heading_features(),
                self._projected_gravity(),
                qvel[:ROOT_QVEL_SIZE],
                qpos[self._joint_qpos_adrs],
                qvel[self._joint_qvel_adrs],
                self._previous_action,
                np.array([math.sin(phase), math.cos(phase), self._command_velocity_x]),
            ]
        )
        return obs.astype(np.float32)

    def _normalized_action(self, action: np.ndarray) -> np.ndarray:
        action_array = np.asarray(action, dtype=np.float64)
        if action_array.shape != (N_ACTUATORS,):
            raise ValueError(f"Expected action shape {(N_ACTUATORS,)}, got {action_array.shape}.")
        return np.clip(action_array, -1.0, 1.0)

    def _command_velocity_from_options(self, options: dict[str, Any] | None) -> float:
        command = self.config.command_velocity_x
        if options is not None and "command_velocity_x" in options:
            command = float(options["command_velocity_x"])
        elif self.config.randomize_command_velocity:
            low, high = self.config.command_velocity_range
            command = float(self.np_random.uniform(low, high))
        low, high = self.config.command_velocity_range
        return float(np.clip(command, low, high))

    def _apply_reset_randomization(self) -> None:
        config = self.config.reset_randomization
        if not config.enabled:
            return

        if config.base_xy_range_m > 0.0:
            self.data.qpos[0:2] += self.np_random.uniform(
                -config.base_xy_range_m,
                config.base_xy_range_m,
                size=2,
            )

        if config.base_yaw_range_rad > 0.0:
            yaw = float(
                self.np_random.uniform(-config.base_yaw_range_rad, config.base_yaw_range_rad)
            )
            half_yaw = 0.5 * yaw
            self.data.qpos[3:7] = np.array(
                [math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)],
                dtype=np.float64,
            )

        if config.joint_position_range_rad > 0.0:
            joint_noise = self.np_random.uniform(
                -config.joint_position_range_rad,
                config.joint_position_range_rad,
                size=N_ACTUATORS,
            )
            self.data.qpos[self._joint_qpos_adrs] = np.clip(
                self.data.qpos[self._joint_qpos_adrs] + joint_noise,
                self._joint_pos_low,
                self._joint_pos_high,
            )

        if config.root_velocity_range > 0.0:
            self.data.qvel[:ROOT_QVEL_SIZE] = self.np_random.uniform(
                -config.root_velocity_range,
                config.root_velocity_range,
                size=ROOT_QVEL_SIZE,
            )

        if config.joint_velocity_range > 0.0:
            self.data.qvel[self._joint_qvel_adrs] = self.np_random.uniform(
                -config.joint_velocity_range,
                config.joint_velocity_range,
                size=N_ACTUATORS,
            )

        if config.randomize_gait_phase and self.config.gait.frequency_hz > 0.0:
            period_s = 1.0 / self.config.gait.frequency_hz
            self.data.time = float(self.np_random.uniform(0.0, period_s))

        mujoco.mj_forward(self.model, self.data)

    def _gait_phase(self, time_s: float) -> float:
        return 2.0 * math.pi * self.config.gait.frequency_hz * time_s

    def _lateral_displacement(self) -> float:
        return float(self.data.qpos[1] - self._episode_start_y)

    def _heading_error(self) -> float:
        return wrap_angle(base_yaw(self.data.qpos))

    def _heading_features(self) -> np.ndarray:
        heading_error = self._heading_error()
        return np.array([math.sin(heading_error), math.cos(heading_error)], dtype=np.float64)

    def _projected_gravity(self) -> np.ndarray:
        quat = np.asarray(self.data.qpos[3:7], dtype=np.float64)
        rot = np.empty(9, dtype=np.float64)
        mujoco.mju_quat2Mat(rot, quat)
        body_to_world = rot.reshape(3, 3)
        gravity_world = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        return np.clip(body_to_world.T @ gravity_world, -1.0, 1.0)

    def _reward(
        self,
        action: np.ndarray,
        action_rate: np.ndarray,
        terminated: bool,
    ) -> tuple[float, dict[str, float]]:
        reward_config = self.config.reward
        forward_velocity = float(self.data.qvel[0])
        velocity_error = forward_velocity - self._command_velocity_x
        velocity_tracking = math.exp(
            -(velocity_error * velocity_error)
            / max(
                reward_config.velocity_tracking_sigma * reward_config.velocity_tracking_sigma,
                1e-8,
            )
        )

        lateral_displacement = self._lateral_displacement()
        straight_line = math.exp(
            -(lateral_displacement * lateral_displacement)
            / max(reward_config.straight_line_sigma * reward_config.straight_line_sigma, 1e-8)
        )
        lateral_velocity_penalty = float(self.data.qvel[1] * self.data.qvel[1])

        heading_error = self._heading_error()
        heading = math.exp(
            -(heading_error * heading_error)
            / max(reward_config.heading_sigma * reward_config.heading_sigma, 1e-8)
        )
        yaw_rate_penalty = float(self.data.qvel[5] * self.data.qvel[5])

        height_error = float(self.data.qpos[2]) - self.config.target_height
        height_reward = math.exp(
            -(height_error * height_error)
            / max(reward_config.height_sigma * reward_config.height_sigma, 1e-8)
        )

        roll, pitch = base_roll_pitch(self.data.qpos)
        roll_pitch_penalty = roll * roll + pitch * pitch
        action_penalty = float(np.mean(np.square(action)))
        action_rate_penalty = float(np.mean(np.square(action_rate)))
        joint_vel = self.data.qvel[self._joint_qvel_adrs]
        joint_force = self.data.qfrc_actuator[self._joint_qvel_adrs]
        energy_penalty = float(np.mean(np.abs(joint_force * joint_vel)))
        alive = 0.0 if terminated else 1.0

        terms = {
            "velocity_tracking": reward_config.velocity_tracking_weight * velocity_tracking,
            "straight_line": reward_config.straight_line_weight * straight_line,
            "lateral_velocity_penalty": (
                -reward_config.lateral_velocity_weight * lateral_velocity_penalty
            ),
            "heading": reward_config.heading_weight * heading,
            "yaw_rate_penalty": -reward_config.yaw_rate_weight * yaw_rate_penalty,
            "alive": reward_config.alive_weight * alive,
            "height": reward_config.height_weight * height_reward,
            "roll_pitch_penalty": -reward_config.roll_pitch_weight * roll_pitch_penalty,
            "action_penalty": -reward_config.action_weight * action_penalty,
            "action_rate_penalty": -reward_config.action_rate_weight * action_rate_penalty,
            "energy_penalty": -reward_config.energy_weight * energy_penalty,
        }
        reward = float(sum(terms.values()))
        return reward, terms

    def _is_terminated(self) -> bool:
        if float(self.data.qpos[2]) < self.config.min_base_height:
            return True
        roll, pitch = base_roll_pitch(self.data.qpos)
        return (
            abs(roll) > self.config.max_abs_roll_pitch_rad
            or abs(pitch) > self.config.max_abs_roll_pitch_rad
        )

    def _is_truncated(self) -> bool:
        elapsed = float(self.data.time) - self._episode_start_time
        return elapsed >= self.config.max_episode_time_s

    def _schedule_next_push(self, now: float) -> None:
        push_config = self.config.random_pushes
        if not push_config.enabled:
            self._next_push_time = math.inf
            return

        interval = float(
            self.np_random.uniform(push_config.min_interval_s, push_config.max_interval_s)
        )
        self._next_push_time = now + interval

    def _update_push(self, now: float) -> None:
        push_config = self.config.random_pushes
        if not push_config.enabled:
            self.physics.clear_push()
            self._push_force.fill(0.0)
            return

        if now < self._push_end_time:
            self.physics.apply_push(self._push_force)
            return

        if self._push_end_time != -math.inf:
            self.physics.clear_push()
            self._push_force.fill(0.0)
            self._push_end_time = -math.inf
            self._schedule_next_push(now)

        if now >= self._next_push_time:
            angle = float(self.np_random.uniform(0.0, 2.0 * math.pi))
            magnitude = float(
                self.np_random.uniform(push_config.min_force_n, push_config.max_force_n)
            )
            duration = float(
                self.np_random.uniform(push_config.min_duration_s, push_config.max_duration_s)
            )
            self._push_force[:] = magnitude * np.array(
                [math.cos(angle), math.sin(angle), 0.0], dtype=np.float64
            )
            self._push_end_time = now + duration
            self.physics.apply_push(self._push_force)

    def _info(self, reward_terms: dict[str, float]) -> dict[str, Any]:
        roll, pitch = base_roll_pitch(self.data.qpos)
        yaw = base_yaw(self.data.qpos)
        return {
            "time": float(self.data.time),
            "command_velocity_x": self._command_velocity_x,
            "forward_velocity": float(self.data.qvel[0]),
            "lateral_velocity": float(self.data.qvel[1]),
            "lateral_displacement": self._lateral_displacement(),
            "base_height": float(self.data.qpos[2]),
            "roll": roll,
            "pitch": pitch,
            "base_yaw": yaw,
            "heading_error": wrap_angle(yaw),
            "yaw_rate": float(self.data.qvel[5]),
            "push_force": self._push_force.copy(),
            "reward_terms": reward_terms,
        }
