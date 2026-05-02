"""
random_baseline.py
------------------
Phase 2 — Step 2: Establish a random-action baseline for HalfCheetah-v5.

This script runs N episodes with a uniformly random policy and records
episodic returns. The baseline gives us the lower bound to compare our
trained SAC agent against in the report.

Run:
    python random_baseline.py

Outputs:
    results/baseline_returns.csv   — per-episode return data
    results/baseline_summary.txt   — mean, std, min, max stats
    results/baseline_plot.png      — return distribution histogram
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym

# ── Configuration ─────────────────────────────────────────────────────────────
ENV_ID       = "HalfCheetah-v5"
N_EPISODES   = 100
SEED         = 42
RESULTS_DIR  = "results"
# ──────────────────────────────────────────────────────────────────────────────


def run_random_baseline(
    env_id: str = ENV_ID,
    n_episodes: int = N_EPISODES,
    seed: int = SEED,
) -> list[float]:
    """
    Roll out a uniformly random policy for n_episodes and return
    the list of episodic returns.
    """
    env = gym.make(env_id)
    returns = []

    print(f"Running {n_episodes} random episodes on {env_id} ...")
    print("-" * 45)

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        episode_return = 0.0
        terminated = truncated = False

        while not (terminated or truncated):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_return += reward

        returns.append(episode_return)

        if (ep + 1) % 10 == 0:
            recent_mean = np.mean(returns[-10:])
            print(f"  Episode {ep + 1:>3}/{n_episodes} | "
                  f"Return: {episode_return:>8.2f} | "
                  f"Last-10 mean: {recent_mean:>8.2f}")

    env.close()
    return returns


def save_results(returns: list[float], results_dir: str = RESULTS_DIR) -> None:
    """Save per-episode returns to CSV and print summary statistics."""
    os.makedirs(results_dir, exist_ok=True)

    # Per-episode CSV
    df = pd.DataFrame({
        "episode": range(1, len(returns) + 1),
        "return":  returns,
    })
    csv_path = os.path.join(results_dir, "baseline_returns.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved returns to: {csv_path}")

    # Summary stats
    summary = {
        "mean":   np.mean(returns),
        "std":    np.std(returns),
        "min":    np.min(returns),
        "max":    np.max(returns),
        "median": np.median(returns),
    }
    txt_path = os.path.join(results_dir, "baseline_summary.txt")
    with open(txt_path, "w") as f:
        f.write(f"Random Baseline Summary — {ENV_ID}\n")
        f.write("=" * 40 + "\n")
        for k, v in summary.items():
            f.write(f"{k:>8}: {v:.2f}\n")

    print("\n[Baseline Summary]")
    print("=" * 40)
    for k, v in summary.items():
        print(f"  {k:>8}: {v:.2f}")

    return summary


def plot_returns(returns: list[float], results_dir: str = RESULTS_DIR) -> None:
    """Plot the return distribution and per-episode curve."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(
        f"Random Baseline — {ENV_ID} ({len(returns)} episodes)",
        fontsize=13, fontweight="bold"
    )

    # Left: per-episode return
    ax1 = axes[0]
    ax1.plot(returns, color="#378ADD", linewidth=0.9, alpha=0.7, label="Episode return")
    ax1.axhline(np.mean(returns), color="#D85A30", linewidth=1.5,
                linestyle="--", label=f"Mean: {np.mean(returns):.1f}")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Return")
    ax1.set_title("Episodic returns")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Right: histogram
    ax2 = axes[1]
    ax2.hist(returns, bins=20, color="#378ADD", alpha=0.8, edgecolor="white", linewidth=0.5)
    ax2.axvline(np.mean(returns), color="#D85A30", linewidth=1.5,
                linestyle="--", label=f"Mean: {np.mean(returns):.1f}")
    ax2.set_xlabel("Return")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Return distribution")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(results_dir, "baseline_plot.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to: {plot_path}")


if __name__ == "__main__":
    returns = run_random_baseline()
    save_results(returns)
    plot_returns(returns)
    print("\n[Done] Baseline complete. Use the mean return as your comparison benchmark.\n")
