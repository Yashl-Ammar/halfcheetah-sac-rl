# AI-832 RL Semester Project
## HalfCheetah-v5 · Soft Actor-Critic (SAC)

---

## Setup

### 1. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 2. Install dependencies
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

> **GPU note:** PyTorch with CUDA is installed automatically via SB3's extras.
> To confirm your RTX 3070 is visible:
> ```python
> import torch
> print(torch.cuda.is_available())       # True
> print(torch.cuda.get_device_name(0))   # NVIDIA GeForce RTX 3070
> ```

---

## Project structure

```
halfcheetah_rl/
├── requirements.txt        # All dependencies
├── verify_env.py           # Phase 2 — confirm environment works
├── random_baseline.py      # Phase 2 — random policy baseline
├── train.py                # Phase 3 — SAC training (coming next)
├── evaluate.py             # Phase 4 — evaluation & analysis (coming next)
├── results/                # Auto-created by scripts
│   ├── baseline_returns.csv
│   ├── baseline_summary.txt
│   └── baseline_plot.png
└── models/                 # Saved checkpoints (created during training)
```

---

## Reproducing results

### Phase 2 — Environment setup & baseline
```bash
# Step 1: Verify installation
python verify_env.py

# Step 2: Run random baseline (100 episodes)
python random_baseline.py
```

### Phase 3 — Training (coming next)
```bash
python train.py
```

### Phase 4 — Evaluation (coming next)
```bash
python evaluate.py
```

---

## Environment details

| Property         | Value                  |
|------------------|------------------------|
| Environment ID   | `HalfCheetah-v5`       |
| Observation dim  | 17                     |
| Action dim       | 6 (continuous, ±1)     |
| Reward           | Dense (velocity − cost)|
| Max episode steps| 1000                   |
| Algorithm        | SAC (Stable-Baselines3)|

---

## Hardware
- GPU: NVIDIA GeForce RTX 3070
- Framework: PyTorch + Stable-Baselines3
