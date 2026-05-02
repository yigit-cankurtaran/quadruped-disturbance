# Quadruped Demo

MuJoCo Unitree Go1 project with scripted visual demos and an RL-ready residual control environment.

The robot model is the Google DeepMind MuJoCo Menagerie Unitree Go1 model under
"models/unitree_go1/". The project keeps the low-level MuJoCo wrapper separate from
the Gymnasium task environment so the simple demos remain useful while RL code can
build on the same simulator.

## Visual Demos

Run the existing scripted controllers:

```bash
uv run mjpython scripts/run_standing.py
uv run mjpython scripts/run_trot.py
```

For headless smoke runs:

```bash
uv run python scripts/run_standing.py --duration 1 --no-viewer
uv run python scripts/run_trot.py --duration 1 --no-viewer
```

Both scripts can apply deterministic lateral pushes:

```bash
uv run mjpython scripts/run_trot.py --pushes
```

## RL-Ready Environment

`quadruped_demo.rl_env.Go1TrotRLEnv` is a Gymnasium-compatible task environment that
wraps `Go1Env`. It exposes the standard:

```python
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(action)
```

The action is a normalized 12D residual in actuator order:

```text
FR hip/thigh/calf, FL hip/thigh/calf, RR hip/thigh/calf, RL hip/thigh/calf
```

At each physics step the env computes:

```python
base_target = TrotGait.target(time)
ctrl = base_target + action_scale * action
```

The resulting control is clipped to the model actuator control ranges. The RL action
is held for "decimation" MuJoCo physics steps, while the open-loop gait phase continues
to advance at the MuJoCo timestep.

The observation is compact and explicit:

- base height
- projected gravity in the body frame
- freejoint root velocity
- 12 actuated joint positions
- 12 actuated joint velocities
- previous residual action
- gait phase sin/cos
- desired forward command velocity

Rewards include forward velocity tracking, alive/upright and height terms, plus
roll/pitch, action magnitude, action rate, and actuator energy penalties. Episodes
terminate on low body height or excessive roll/pitch, and truncate at the configured
episode time.

Random push disturbances are available through "RandomPushConfig" for training or
evaluation. The original visual demo push behavior remains in "run_controller".

## Random RL Rollout

Run the Gymnasium env with random residual actions:

```bash
uv run python scripts/run_rl_random.py --duration 1 --no-viewer
```

With viewer and randomized pushes:

```bash
uv run mjpython scripts/run_rl_random.py --pushes
```

## Development

```bash
uv run ruff check .
uv run pytest
```

