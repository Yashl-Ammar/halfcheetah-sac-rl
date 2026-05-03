"""
robustness_eval.py
------------------
Evaluates the trained SAC agent robustness on HalfCheetah-v5.

  1. Multi-seed evaluation   — 3 seeds × 20 episodes
  2. Control cost variation  — 3 ctrl_cost_weight values × 20 episodes

Outputs:
  results/robustness_seeds.txt
  results/robustness_ctrl_cost.txt
  results/plot_robustness.png
"""

import os

import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

plt.style.use("seaborn-v0_8-whitegrid")

MODEL_PATH  = os.path.join("results", "sac_eval_run1", "best_model.zip")
RESULTS_DIR = "results"
SAC_COLOR   = "#378ADD"
ENV_ID      = "HalfCheetah-v5"
SEEDS       = [42, 123, 456]
CTRL_COSTS  = [0.05, 0.10, 0.20]
N_EPISODES  = 20


# ── Core evaluation ────────────────────────────────────────────────────────────
def run_episodes(model, env, n_episodes: int, seed: int | None = None) -> np.ndarray:
    """Run n_episodes with deterministic policy. Seeds the first reset only."""
    returns = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed if ep == 0 else None)
        done, total = False, 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total += reward
        returns.append(total)
        print(f"    ep {ep + 1:2d}/{n_episodes}  return={total:10.2f}", flush=True)
    return np.array(returns)


# ── 1. Multi-seed ──────────────────────────────────────────────────────────────
def eval_seeds(model) -> dict[int, np.ndarray]:
    results = {}
    for seed in SEEDS:
        print(f"\n[Seed {seed}]")
        env = Monitor(gym.make(ENV_ID))
        results[seed] = run_episodes(model, env, N_EPISODES, seed=seed)
        env.close()
    return results


def save_seed_table(seed_returns: dict[int, np.ndarray]) -> None:
    all_r = np.concatenate(list(seed_returns.values()))
    W = 60
    rows = [
        "=" * W,
        "  Multi-seed Evaluation — SAC on HalfCheetah-v5",
        f"  {N_EPISODES} episodes per seed",
        "=" * W,
        f"  {'Seed':<12} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}",
        "-" * W,
    ]
    for seed, r in seed_returns.items():
        rows.append(f"  {seed:<12} {np.mean(r):>10.2f} {np.std(r):>10.2f} "
                    f"{np.min(r):>10.2f} {np.max(r):>10.2f}")
    rows += [
        "-" * W,
        f"  {'Overall':<12} {np.mean(all_r):>10.2f} {np.std(all_r):>10.2f} "
        f"{np.min(all_r):>10.2f} {np.max(all_r):>10.2f}",
        "=" * W,
    ]
    _write_table(os.path.join(RESULTS_DIR, "robustness_seeds.txt"), rows)


# ── 2. Control cost variation ──────────────────────────────────────────────────
def eval_ctrl_costs(model) -> dict[float, np.ndarray]:
    results = {}
    for cost in CTRL_COSTS:
        print(f"\n[ctrl_cost_weight={cost}]")
        env = Monitor(gym.make(ENV_ID, ctrl_cost_weight=cost))
        results[cost] = run_episodes(model, env, N_EPISODES, seed=42)
        env.close()
    return results


def save_ctrl_cost_table(cost_returns: dict[float, np.ndarray]) -> None:
    W = 65
    rows = [
        "=" * W,
        "  Control Cost Variation — SAC on HalfCheetah-v5",
        f"  {N_EPISODES} episodes per condition",
        "=" * W,
        f"  {'ctrl_cost_weight':<22} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}",
        "-" * W,
    ]
    for cost, r in cost_returns.items():
        tag   = " (default)" if cost == 0.10 else ""
        label = f"{cost:.2f}{tag}"
        rows.append(f"  {label:<22} {np.mean(r):>10.2f} {np.std(r):>10.2f} "
                    f"{np.min(r):>10.2f} {np.max(r):>10.2f}")
    rows.append("=" * W)
    _write_table(os.path.join(RESULTS_DIR, "robustness_ctrl_cost.txt"), rows)


def _write_table(path: str, rows: list[str]) -> None:
    text = "\n".join(rows)
    print(text)
    with open(path, "w") as f:
        f.write(text + "\n")
    print(f"\n  -> Saved: {path}\n")


# ── 3. Plot ────────────────────────────────────────────────────────────────────
def plot_robustness(seed_returns: dict[int, np.ndarray],
                    cost_returns: dict[float, np.ndarray],
                    out_path: str) -> None:
    fig, (ax_box, ax_bar) = plt.subplots(1, 2, figsize=(12, 5))

    # ── Left: box plot + scatter overlay ──────────────────────────────────────
    data   = [seed_returns[s] for s in SEEDS]
    labels = [f"Seed {s}" for s in SEEDS]

    bp = ax_box.boxplot(
        data, labels=labels, patch_artist=True, widths=0.5, showfliers=False,
        medianprops=dict(color="white", linewidth=2.0),
        whiskerprops=dict(color=SAC_COLOR, linewidth=1.2),
        capprops=dict(color=SAC_COLOR, linewidth=1.2),
        boxprops=dict(linewidth=0),
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(SAC_COLOR)
        patch.set_alpha(0.55)

    rng = np.random.default_rng(0)
    for i, returns in enumerate(data):
        jitter = rng.normal(0, 0.06, len(returns))
        ax_box.scatter(i + 1 + jitter, returns,
                       color=SAC_COLOR, alpha=0.55, s=22, zorder=3)

    ax_box.set_ylabel("Episodic return")
    ax_box.set_title("Multi-seed", fontweight="bold")

    # ── Right: bar chart ───────────────────────────────────────────────────────
    costs  = list(cost_returns.keys())
    means  = [np.mean(cost_returns[c]) for c in costs]
    stds   = [np.std(cost_returns[c])  for c in costs]
    x_pos  = np.arange(len(costs))

    ax_bar.bar(x_pos, means, yerr=stds, color=SAC_COLOR, alpha=0.80, width=0.5,
               capsize=5, error_kw=dict(lw=1.5, capthick=1.5))
    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels([str(c) for c in costs])
    ax_bar.set_xlabel("ctrl_cost_weight")
    ax_bar.set_ylabel("Mean episodic return")
    ax_bar.set_title("Control cost variation", fontweight="bold")

    fig.suptitle("Robustness evaluation — SAC on HalfCheetah-v5",
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"Loading model from {MODEL_PATH} ...")
    model = SAC.load(MODEL_PATH, device="cpu")

    print("\n=== 1 / 2  Multi-seed evaluation ===")
    seed_returns = eval_seeds(model)
    save_seed_table(seed_returns)

    print("=== 2 / 2  Control cost variation ===")
    cost_returns = eval_ctrl_costs(model)
    save_ctrl_cost_table(cost_returns)

    plot_robustness(
        seed_returns, cost_returns,
        os.path.join(RESULTS_DIR, "plot_robustness.png"),
    )
