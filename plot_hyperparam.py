"""
plot_hyperparam.py
------------------
Produces two hyperparameter sensitivity plots from per-run reward CSVs.

Run:
    python plot_hyperparam.py

Outputs:
    results/plot_hyperparam_all.png
    results/plot_hyperparam_grouped.png
"""

import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

plt.style.use("seaborn-v0_8-whitegrid")

RESULTS_DIR     = "results"
WINDOW          = 30
X_CLIP          = 500_000
RANDOM_BASELINE = -292

# ── Run definitions ────────────────────────────────────────────────────────────
RUNS = [
    dict(csv="reward_log_run1.csv",
         label="Default (lr=3e-4, γ=0.99, b=256)", color="#378ADD", lw=2.5),
    dict(csv="reward_log_lr_low.csv",
         label="lr = 1e-4",    color="#D85A30", lw=1.8),
    dict(csv="reward_log_lr_high.csv",
         label="lr = 1e-3",    color="#E8A838", lw=1.8),
    dict(csv="reward_log_gamma_low.csv",
         label="γ = 0.95",     color="#9B59B6", lw=1.8),
    dict(csv="reward_log_gamma_high.csv",
         label="γ = 0.999",    color="#2ECC71", lw=1.8),
    dict(csv="reward_log_batch_small.csv",
         label="batch = 128",  color="#E74C3C", lw=1.8),
    dict(csv="reward_log_batch_large.csv",
         label="batch = 512",  color="#1ABC9C", lw=1.8),
]

GROUPS = [
    ("Learning rate", ["reward_log_run1.csv", "reward_log_lr_low.csv",      "reward_log_lr_high.csv"]),
    ("Discount γ",    ["reward_log_run1.csv", "reward_log_gamma_low.csv",   "reward_log_gamma_high.csv"]),
    ("Batch size",    ["reward_log_run1.csv", "reward_log_batch_small.csv", "reward_log_batch_large.csv"]),
]

_RUN_BY_CSV = {r["csv"]: r for r in RUNS}


# ── Helpers ────────────────────────────────────────────────────────────────────
def _load_smoothed(csv_name: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Load a reward CSV, clip to X_CLIP steps, smooth with WINDOW. Returns None if file missing."""
    path = os.path.join(RESULTS_DIR, csv_name)
    if not os.path.exists(path):
        print(f"  [warn] {path} not found — skipping")
        return None
    data = np.genfromtxt(path, delimiter=",", names=True)
    ts   = data["timestep"].astype(np.int64)
    rw   = data["reward"]
    ts, rw = ts[ts <= X_CLIP], rw[ts <= X_CLIP]
    if len(rw) < WINDOW:
        return ts, rw
    sm = np.convolve(rw, np.ones(WINDOW) / WINDOW, mode="valid")
    return ts[WINDOW - 1:], sm


def _plot_run(ax: plt.Axes, csv_name: str) -> bool:
    """Plot one smoothed run onto ax. Returns True if data was found."""
    result = _load_smoothed(csv_name)
    if result is None:
        return False
    ts, sm = result
    cfg = _RUN_BY_CSV[csv_name]
    ax.plot(ts, sm, label=cfg["label"], color=cfg["color"], linewidth=cfg["lw"])
    return True


def _decorate(ax: plt.Axes, title: str = "", show_ylabel: bool = True) -> None:
    ax.axhline(RANDOM_BASELINE, color="grey", linestyle="--",
               linewidth=1.2, alpha=0.7, label=f"Random baseline ({RANDOM_BASELINE})")
    ax.set_xlim(0, X_CLIP)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}K"))
    ax.set_xlabel("Environment steps")
    if show_ylabel:
        ax.set_ylabel("Episodic return (smoothed)")
    if title:
        ax.set_title(title, fontweight="bold")


# ── Plot 1: all 7 runs overlaid ────────────────────────────────────────────────
def plot_all(out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(13, 6))

    for run in RUNS:
        _plot_run(ax, run["csv"])

    _decorate(ax, title="Hyperparameter sensitivity — SAC on HalfCheetah-v5")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5),
              frameon=True, fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ── Plot 2: one subplot per hyperparameter group ───────────────────────────────
def plot_grouped(out_path: str) -> None:
    fig, axes = plt.subplots(1, len(GROUPS), figsize=(16, 5))

    for i, (ax, (group_title, csv_names)) in enumerate(zip(axes, GROUPS)):
        for csv_name in csv_names:
            _plot_run(ax, csv_name)
        _decorate(ax, title=group_title, show_ylabel=(i == 0))
        ax.legend(fontsize=8.5, frameon=True)

    fig.suptitle("Hyperparameter sensitivity by group — SAC on HalfCheetah-v5",
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plot_all(    os.path.join(RESULTS_DIR, "plot_hyperparam_all.png"))
    plot_grouped(os.path.join(RESULTS_DIR, "plot_hyperparam_grouped.png"))
