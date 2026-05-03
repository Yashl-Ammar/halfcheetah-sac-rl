"""
analyze.py
----------
Phase 4 analysis: produces all plots and the summary table for a SAC
HalfCheetah-v5 training run.

Run:
    python analyze.py [--run RUN_NAME]

Outputs (all written to results/):
    plot_convergence.png
    plot_baseline_comparison.png
    plot_training_metrics.png
    analysis_summary.txt
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# ── Style ──────────────────────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-whitegrid")

# ── Colors ─────────────────────────────────────────────────────────────────────
SAC_COLOR  = "#378ADD"
BASE_COLOR = "#D85A30"

# ── Hardcoded statistics (from the completed run1 + random_baseline.py) ────────
RANDOM_MEAN    = -292.10
RANDOM_STD     =   78.02
RANDOM_MIN     = -603.49
RANDOM_MAX     =  -91.36

SAC_FINAL_MEAN = 12_558.03
SAC_FINAL_STD  =    964.02
SAC_BEST_MEAN  = 13_164.53
SAC_BEST_STD   =     37.23


# ── Helpers ────────────────────────────────────────────────────────────────────
def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    return np.convolve(values, np.ones(window) / window, mode="valid")


def _step_formatter(x, _):
    return f"{x / 1e6:.1f}M"


# ── 1. Convergence plot ────────────────────────────────────────────────────────
def plot_convergence(csv_path: str, out_path: str, window: int = 50) -> None:
    data      = np.genfromtxt(csv_path, delimiter=",", names=True)
    timesteps = data["timestep"].astype(np.int64)
    rewards   = data["reward"]

    smoothed  = _smooth(rewards, window)
    smooth_ts = timesteps[window - 1:]

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(timesteps, rewards,
            color=SAC_COLOR, alpha=0.20, linewidth=0.7, label="Episode return")
    ax.plot(smooth_ts, smoothed,
            color=SAC_COLOR, linewidth=2.0, label=f"Smoothed (window={window})")
    ax.axhline(RANDOM_MEAN, color=BASE_COLOR, linestyle="--", linewidth=1.5,
               label=f"Random baseline ({RANDOM_MEAN:.0f})")

    _annotate_crossing(ax, smooth_ts, smoothed, threshold=0,     label="First ≥ 0",
                       y_offset=1_200)
    _annotate_crossing(ax, smooth_ts, smoothed, threshold=5_000, label="First ≥ 5 000",
                       y_offset=1_200)

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_step_formatter))
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Episodic return")
    ax.set_title("SAC Convergence — HalfCheetah-v5", fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def _annotate_crossing(ax, smooth_ts, smoothed, threshold, label, y_offset):
    mask = smoothed >= threshold
    if not mask.any():
        return
    idx  = int(np.argmax(mask))
    x, y = float(smooth_ts[idx]), float(smoothed[idx])

    # Place text left of the point if it's in the final 25% of the run
    right_edge = float(smooth_ts[-1])
    if x > 0.75 * right_edge:
        x_text, ha = x - right_edge * 0.06, "right"
    else:
        x_text, ha = x + right_edge * 0.04, "left"

    ax.scatter([x], [y], color="black", zorder=5, s=40)
    ax.annotate(
        f"{label}\n({x / 1e6:.2f}M steps)",
        xy=(x, y),
        xytext=(x_text, y + y_offset),
        ha=ha,
        fontsize=8.5,
        arrowprops=dict(arrowstyle="->", color="black", lw=1.0),
    )


# ── 2. Baseline comparison ─────────────────────────────────────────────────────
def plot_baseline_comparison(out_path: str) -> None:
    labels = ["Random policy", "SAC final-100", "SAC best eval"]
    means  = [RANDOM_MEAN,  SAC_FINAL_MEAN, SAC_BEST_MEAN]
    stds   = [RANDOM_STD,   SAC_FINAL_STD,  SAC_BEST_STD]
    colors = [BASE_COLOR,   SAC_COLOR,       SAC_COLOR]

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.barh(labels, means, xerr=stds,
                   color=colors, alpha=0.85, capsize=5,
                   error_kw=dict(lw=1.5, capthick=1.5))

    x_span = max(means) - min(means)
    pad    = x_span * 0.012

    for bar, mean, std in zip(bars, means, stds):
        if mean >= 0:
            x_label, ha = mean + std + pad, "left"
        else:
            x_label, ha = mean - std - pad, "right"
        ax.text(x_label, bar.get_y() + bar.get_height() / 2,
                f"{mean:,.0f}", va="center", ha=ha,
                fontsize=10, fontweight="bold")

    ax.axvline(0, color="grey", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Mean episodic return")
    ax.set_title("Policy comparison — HalfCheetah-v5", fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ── 3. Training metrics from TensorBoard ──────────────────────────────────────
def plot_training_metrics(log_dir: str, out_path: str, window: int = 30) -> None:
    if not os.path.isdir(log_dir):
        print(f"TensorBoard log dir not found: {log_dir} — skipping metrics plot.")
        return

    ea = EventAccumulator(log_dir, size_guidance={"scalars": 0})
    ea.Reload()
    available = set(ea.Tags().get("scalars", []))

    wanted = [
        ("train/actor_loss",  "Actor loss"),
        ("train/critic_loss", "Critic loss"),
        ("train/ent_coef",    "Entropy coef"),
    ]
    metrics = [(tag, lbl) for tag, lbl in wanted if tag in available]

    if not metrics:
        print(f"None of the expected tags found in {log_dir}.")
        print(f"  Available scalars: {sorted(available)}")
        return

    fig, axes = plt.subplots(len(metrics), 1,
                             figsize=(12, 3.5 * len(metrics)),
                             sharex=True)
    if len(metrics) == 1:
        axes = [axes]

    for ax, (tag, ylabel) in zip(axes, metrics):
        events = ea.Scalars(tag)
        steps  = np.array([e.step  for e in events])
        values = np.array([e.value for e in events])

        ax.plot(steps, values, color=SAC_COLOR, alpha=0.20, linewidth=0.7)
        if len(values) >= window:
            sm = _smooth(values, window)
            ax.plot(steps[window - 1:], sm,
                    color=SAC_COLOR, linewidth=2.0, label=f"Smoothed (w={window})")
        ax.set_ylabel(ylabel)

        y_lo = np.percentile(values, 1)
        y_hi = np.percentile(values, 99)
        margin = (y_hi - y_lo) * 0.05
        ax.set_ylim(y_lo - margin, y_hi + margin)

    axes[-1].xaxis.set_major_formatter(mticker.FuncFormatter(_step_formatter))
    axes[-1].set_xlabel("Environment steps")
    fig.suptitle("SAC Training Metrics — HalfCheetah-v5",
                 fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ── 4. Summary table ───────────────────────────────────────────────────────────
def save_summary(csv_path: str, out_path: str) -> None:
    data    = np.genfromtxt(csv_path, delimiter=",", names=True)
    rewards = data["reward"]
    final   = rewards[-100:] if len(rewards) >= 100 else rewards

    sac_mean = float(np.mean(final))
    sac_std  = float(np.std(final))
    sac_max  = float(np.max(rewards))

    abs_gain = sac_mean - RANDOM_MEAN
    pct_gain = abs_gain / abs(RANDOM_MEAN) * 100

    W = 50
    lines = [
        "=" * W,
        "  Phase 4 Analysis Summary — HalfCheetah-v5",
        "=" * W,
        "",
        "[Random baseline]",
        f"  Mean return        : {RANDOM_MEAN:>10.2f}",
        f"  Std return         : {RANDOM_STD:>10.2f}",
        f"  Min return         : {RANDOM_MIN:>10.2f}",
        f"  Max return         : {RANDOM_MAX:>10.2f}",
        "",
        "[SAC trained]",
        f"  Final-100 mean     : {sac_mean:>10.2f}",
        f"  Final-100 std      : {sac_std:>10.2f}",
        f"  Max episode return : {sac_max:>10.2f}",
        f"  Best eval mean     : {SAC_BEST_MEAN:>10.2f}",
        f"  Best eval std      : {SAC_BEST_STD:>10.2f}",
        "",
        "[Improvement over random]",
        f"  Absolute           : {abs_gain:>+10.2f}",
        f"  Relative           : {pct_gain:>+9.1f}%",
        "=" * W,
    ]
    text = "\n".join(lines)
    print(text)
    with open(out_path, "w") as f:
        f.write(text + "\n")
    print(f"\nSaved: {out_path}")


# ── Entry point ────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 4 analysis for a SAC training run")
    p.add_argument("--run", default="run1", help="Run name (default: run1)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run  = args.run

    os.makedirs("results", exist_ok=True)

    csv_path = os.path.join("results", f"reward_log_{run}.csv")
    log_dir  = os.path.join("logs", f"SAC_{run}_1")

    plot_convergence(
        csv_path,
        os.path.join("results", "plot_convergence.png"),
    )
    plot_baseline_comparison(
        os.path.join("results", "plot_baseline_comparison.png"),
    )
    plot_training_metrics(
        log_dir,
        os.path.join("results", "plot_training_metrics.png"),
    )
    save_summary(
        csv_path,
        os.path.join("results", "analysis_summary.txt"),
    )
