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
- lateral displacement from the episode start line
- heading error sin/cos relative to world +X
- projected gravity in the body frame
- freejoint root velocity
- 12 actuated joint positions
- 12 actuated joint velocities
- previous residual action
- gait phase sin/cos
- desired forward command velocity

Rewards include forward velocity tracking, straight-line tracking, heading alignment,
alive/upright and height terms, plus roll/pitch, lateral velocity, yaw rate, action
magnitude, action rate, and actuator energy penalties. Episodes terminate on low body
height or excessive roll/pitch, and truncate at the configured episode time.

Random push disturbances are available through "RandomPushConfig" for training or
evaluation. The original visual demo push behavior remains in "run_controller".

Training defaults add mild reset randomization and command velocity sampling so PPO
does not only see one initial state and one requested speed. Push disturbances are
available as a training flag instead of being forced on from the first run.

## Random RL Rollout

Run the Gymnasium env with random residual actions:

```bash
uv run python scripts/run_rl_random.py --duration 1 --no-viewer
```

With viewer and randomized pushes:

```bash
uv run mjpython scripts/run_rl_random.py --pushes
```

## PPO Training

Stable-Baselines3 PPO is wired through `quadruped_demo.training` and the CLI scripts
below. Training uses 4 vectorized envs by default. The default vector backend is
`auto`: it uses `SubprocVecEnv` for multi-env training and `DummyVecEnv` for a single
env. Use `--vec-env dummy` to force simpler single-process stepping. A short smoke run:

```bash
uv run python scripts/train_ppo.py --total-timesteps 1024 --n-steps 128 --batch-size 64
```

The default output directory is `results/ppo/go1_ppo/` and contains:

- `model.zip` and `vecnormalize.pkl` for the final completed model
- `best/best_model.zip` and `best/best_vecnormalize.pkl` from periodic evaluation
- `checkpoints/` for periodic model and VecNormalize snapshots
- monitor logs

If training is interrupted with Ctrl-C, `interrupted_model.zip` and
`interrupted_vecnormalize.pkl` are saved before the script exits.

Watch the default trained checkpoint. The script uses `best/best_model.zip` when it
exists and falls back to the final `model.zip` otherwise:

```bash
uv run mjpython scripts/watch_ppo.py
```

When `--pushes` is enabled, the watch viewer draws a red arrow into the trunk while
a randomized push disturbance is active:

```bash
uv run mjpython scripts/watch_ppo.py --pushes
```

Evaluate a trained checkpoint headlessly:

```bash
uv run python scripts/evaluate_ppo.py \
  --model results/ppo/go1_ppo/model.zip \
  --vecnormalize results/ppo/go1_ppo/vecnormalize.pkl \
  --duration 5 \
  --no-viewer
```

For visual evaluation, use `mjpython` and omit `--no-viewer`:

```bash
uv run mjpython scripts/evaluate_ppo.py \
  --model results/ppo/go1_ppo/model.zip \
  --vecnormalize results/ppo/go1_ppo/vecnormalize.pkl
```

Add `--pushes` to train or evaluate with randomized push disturbances.

## Development

```bash
uv run ruff check .
uv run pytest
```
