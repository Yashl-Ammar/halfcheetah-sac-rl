# AI-832 RL Semester Project
## HalfCheetah-v5 · Soft Actor-Critic (SAC)

---

## Results at a glance

| Metric | Value |
|---|---|
| Algorithm | SAC (Stable-Baselines3) |
| Training steps | 2,000,000 |
| Training time | ~6.6 hours (RTX 3070) |
| Random baseline mean | −292 ± 78 |
| SAC final-100 mean | **12,558 ± 964** |
| SAC best eval mean | **13,165 ± 37** |
| Improvement over random | +12,850 (+4,399%) |
| Multi-seed mean (3 seeds) | 13,380 ± 318 |

---

## Setup

### 1. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 2. Install dependencies
```bash
# PyTorch with CUDA must be installed first
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 3. Verify installation
```bash
python verify_env.py
```

> **GPU check:**
> ```python
> import torch
> print(torch.cuda.is_available())       # True
> print(torch.cuda.get_device_name(0))   # NVIDIA GeForce RTX 3070
> ```

---

## Project structure

```
RL_project/
├── train_sac.py             # Phase 3 — SAC training with auto-resume
├── analyze.py               # Phase 4 — convergence, metrics, summary plots
├── plot_hyperparam.py       # Phase 4 — hyperparameter sensitivity plots
├── robustness_eval.py       # Phase 4 — multi-seed & ctrl_cost evaluation
├── random_baseline.py       # Phase 2 — random policy baseline
├── verify_env.py            # Phase 2 — environment & GPU verification
├── run_experiments.bat      # Batch script for hyperparameter sweep (Windows)
├── requirements.txt
│
├── checkpoints/             # Auto-created by train_sac.py
│   ├── sac_run1/            #   Snapshots every 25K steps + sac_final.zip
│   ├── sac_lr_low/
│   ├── sac_lr_high/
│   ├── sac_gamma_low/
│   ├── sac_gamma_high/
│   ├── sac_batch_small/
│   └── sac_batch_large/
│
├── logs/                    # TensorBoard event files
│   ├── SAC_run1_1/
│   └── SAC_<name>_1/        # One directory per named run
│
└── results/                 # All outputs — auto-created by scripts
    ├── reward_log_run1.csv              # Per-episode returns (all runs)
    ├── reward_log_<name>.csv
    ├── training_curve_<name>.png        # Smoothed reward vs steps
    ├── training_summary_<name>.txt      # Hyperparams + final stats
    ├── sac_eval_<name>/                 # Best model from EvalCallback
    │   └── best_model.zip
    ├── plot_convergence.png             # Phase 4 analysis plots
    ├── plot_baseline_comparison.png
    ├── plot_training_metrics.png
    ├── plot_hyperparam_all.png
    ├── plot_hyperparam_grouped.png
    ├── plot_robustness.png
    ├── analysis_summary.txt
    ├── robustness_seeds.txt
    └── robustness_ctrl_cost.txt
```

---

## Reproducing results

### Phase 2 — Baseline

```bash
# Verify environment and MuJoCo installation
python verify_env.py

# Run random policy for 100 episodes → results/baseline_*.{csv,txt,png}
python random_baseline.py
```

### Phase 3 — Training

```bash
# Full run (2M steps, ~6.6 h on RTX 3070)
python train_sac.py --name run1

# Override any hyperparameter
python train_sac.py --name run1 --timesteps 500000 --lr 3e-4 --seed 42

# Resume automatically (finds the latest checkpoint in checkpoints/sac_run1/)
python train_sac.py --name run1

# Resume from a specific checkpoint
python train_sac.py --name run1 --checkpoint checkpoints/sac_run1/sac_ckpt_1000000_steps.zip
```

Auto-resume behaviour: on every invocation `train_sac.py` scans
`checkpoints/sac_{name}/` for `sac_ckpt_*_steps.zip` files, loads the one with
the highest step count, and continues with `reset_num_timesteps=False` so the
TensorBoard curve and episode counter are uninterrupted.

### Phase 3 — Hyperparameter sweep

```bash
# Windows — runs all 6 variants sequentially (500K steps each)
run_experiments.bat

# Or individually
python train_sac.py --timesteps 500000 --lr 1e-4      --name lr_low
python train_sac.py --timesteps 500000 --lr 1e-3      --name lr_high
python train_sac.py --timesteps 500000 --gamma 0.95   --name gamma_low
python train_sac.py --timesteps 500000 --gamma 0.999  --name gamma_high
python train_sac.py --timesteps 500000 --batch_size 128 --name batch_small
python train_sac.py --timesteps 500000 --batch_size 512 --name batch_large
```

### Phase 4 — Analysis

```bash
# Convergence, baseline comparison, TensorBoard metrics, summary table
python analyze.py --run run1

# Hyperparameter sensitivity (requires all sweep runs to be complete)
python plot_hyperparam.py

# Robustness: multi-seed + control cost variation (requires best_model.zip)
python robustness_eval.py

# Launch TensorBoard
tensorboard --logdir logs
```

---

## Script reference

| Script | Inputs | Key outputs |
|---|---|---|
| `train_sac.py` | CLI flags | checkpoints, reward CSV, training curve |
| `analyze.py` | `results/reward_log_run1.csv`, `logs/SAC_run1_1/` | 3 plots + summary |
| `plot_hyperparam.py` | `results/reward_log_*.csv` | 2 sensitivity plots |
| `robustness_eval.py` | `results/sac_eval_run1/best_model.zip` | 2 tables + plot |
| `random_baseline.py` | — | baseline CSV, plot, summary |
| `verify_env.py` | — | prints env/GPU info |

### `train_sac.py` flags

| Flag | Default | Description |
|---|---|---|
| `--name` | `seed42` | Run name; scopes all output directories |
| `--timesteps` | 2,000,000 | Total environment steps |
| `--seed` | 42 | Random seed |
| `--lr` | 3e-4 | Learning rate |
| `--batch_size` | 256 | Replay buffer batch size |
| `--gamma` | 0.99 | Discount factor |
| `--checkpoint` | *(none)* | Path to `.zip` to resume from explicitly |

---

## SAC hyperparameters (default run)

| Parameter | Value |
|---|---|
| Policy | MlpPolicy [256, 256] |
| Learning rate | 3e-4 |
| Replay buffer | 1,000,000 |
| Learning starts | 10,000 |
| Batch size | 256 |
| τ (soft update) | 0.005 |
| γ (discount) | 0.99 |
| Entropy coef | auto |
| Checkpoint freq | 25,000 steps |
| Eval freq | 25,000 steps |

---

## Key results

### Convergence (run1, 2M steps)

| Metric | Value |
|---|---|
| Total episodes | 2,000 |
| Final-100 mean return | 12,558 ± 964 |
| Best eval mean return | 13,165 ± 37 |
| All-time max episode | 12,991 |

### Hyperparameter sensitivity (500K steps each)

| Variant | Change | Final-100 mean |
|---|---|---|
| Default | lr=3e-4, γ=0.99, b=256 | ~9,200 *(at 500K steps)* |
| lr = 1e-4 | Slower learning | 5,527 |
| lr = 1e-3 | Faster learning | 9,898 |
| γ = 0.95 | Shorter horizon | 7,207 |
| γ = 0.999 | Longer horizon | 7,790 |
| batch = 128 | Smaller batch | 8,944 |
| batch = 512 | Larger batch | 8,764 |

`lr = 1e-3` converges fastest; `lr = 1e-4` is the weakest performer at 500K steps.

### Robustness

**Multi-seed** (best model, 20 episodes × 3 seeds):

| Seed | Mean | Std |
|---|---|---|
| 42 | 13,431 | 133 |
| 123 | 13,320 | 512 |
| 456 | 13,387 | 133 |
| **Overall** | **13,380** | **318** |

**Control cost variation** (20 episodes each):

| ctrl_cost_weight | Mean | Std |
|---|---|---|
| 0.05 (easier) | 13,613 | 133 |
| 0.10 (default) | 13,431 | 133 |
| 0.20 (harder) | 13,068 | 132 |

The trained policy is robust to all three conditions — performance degrades only
~2.6% from the easiest to the hardest control cost setting.

---

## Environment details

| Property | Value |
|---|---|
| Environment ID | `HalfCheetah-v5` |
| Observation space | Box(17,) |
| Action space | Box(6,) continuous ±1 |
| Reward | Forward velocity − ctrl_cost_weight × control² |
| Max episode steps | 1,000 |
| Physics engine | MuJoCo |

---

## Hardware

| Component | Spec |
|---|---|
| GPU | NVIDIA GeForce RTX 3070 (8 GB) |
| PyTorch | 2.5.1+cu121 |
| Stable-Baselines3 | 2.8.0 |
| Gymnasium | ≥ 0.29.0 |

---

## Known issues

**SB3 checkpoint loading on PyTorch 2.5+**

SB3 2.8.0 passes a non-seekable `ZipExtFile` directly to `torch.load()`.
PyTorch 2.5's `PyTorchFileReader` requires a seekable object, causing:

```
RuntimeError: PytorchStreamReader failed reading file .data/serialization_id
```

**Fix** — patch one line in `venv/Lib/site-packages/stable_baselines3/common/save_util.py`:

```python
# Line 442 — before
th_object = th.load(param_file, map_location=device, weights_only=True)

# After
th_object = th.load(io.BytesIO(param_file.read()), map_location=device, weights_only=True)
```

This patch must be reapplied after reinstalling or upgrading SB3.
