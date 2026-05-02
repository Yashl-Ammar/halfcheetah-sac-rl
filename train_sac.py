"""
train_sac.py
------------
Phase 3: Train a SAC agent on HalfCheetah-v5 using Stable-Baselines3.

Features:
  - TensorBoard logging  (logs/sac_halfcheetah/)
  - Periodic model checkpoints  (checkpoints/)
  - EvalCallback every EVAL_FREQ steps  (results/sac_eval/)
  - Hyperparameters defined in the CONFIG block below

Run:
    python train_sac.py

Optional flags:
    --timesteps  INT   Override TOTAL_TIMESTEPS (e.g. --timesteps 2000000)
    --seed       INT   Override SEED
    --name       STR   Suffix appended to log/checkpoint dirs

Outputs:
    logs/sac_halfcheetah_<name>/          TensorBoard event files
    checkpoints/sac_<name>/               Model snapshots every CHECKPOINT_FREQ steps
    results/sac_eval_<name>/              Best model from EvalCallback
    results/training_curve_<name>.png     Smoothed reward-vs-steps plot
    results/training_summary_<name>.txt   Final evaluation statistics
"""

import argparse
import os
import time

import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor

# ── Hyperparameter Configuration ──────────────────────────────────────────────
CONFIG = {
    # Environment
    "env_id":           "HalfCheetah-v5",
    "seed":             42,

    # Training budget
    "total_timesteps":  2_000_000,

    # SAC core hyperparameters (SB3 defaults are strong; tune for experiments)
    "learning_rate":    3e-4,
    "buffer_size":      1_000_000,       # replay buffer capacity
    "learning_starts":  10_000,          # steps before first gradient update
    "batch_size":       256,
    "tau":              0.005,           # soft target update coefficient
    "gamma":            0.99,            # discount factor
    "train_freq":       1,               # update every N environment steps
    "gradient_steps":   1,               # gradient updates per env step
    "ent_coef":         "auto",          # entropy coefficient (auto-tuned)
    "target_entropy":   "auto",          # -dim(A) when "auto"
    "use_sde":          False,           # gSDE exploration (False = standard)

    # Network architecture
    "policy":           "MlpPolicy",
    "net_arch":         [256, 256],      # hidden layer sizes for actor & critic

    # Callbacks
    "eval_freq":        10_000,          # EvalCallback frequency (env steps)
    "eval_episodes":    10,              # episodes per evaluation
    "checkpoint_freq":  50_000,          # CheckpointCallback frequency

    # Paths
    "log_dir":          "logs",
    "checkpoint_dir":   "checkpoints",
    "results_dir":      "results",
}
# ──────────────────────────────────────────────────────────────────────────────


class RewardLoggerCallback(BaseCallback):
    """Collects mean episodic returns from the Monitor wrapper for plotting."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards: list[float] = []
        self.episode_timesteps: list[int] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            ep_info = info.get("episode")
            if ep_info:
                self.episode_rewards.append(ep_info["r"])
                self.episode_timesteps.append(self.num_timesteps)
        return True


def make_env(env_id: str, seed: int, rank: int = 0) -> gym.Env:
    env = gym.make(env_id)
    env = Monitor(env)
    env.reset(seed=seed + rank)
    return env


def build_callbacks(cfg: dict, run_name: str) -> tuple[CallbackList, RewardLoggerCallback]:
    checkpoint_path = os.path.join(cfg["checkpoint_dir"], f"sac_{run_name}")
    eval_path = os.path.join(cfg["results_dir"], f"sac_eval_{run_name}")

    eval_env = Monitor(gym.make(cfg["env_id"]))
    eval_env.reset(seed=cfg["seed"] + 999)

    checkpoint_cb = CheckpointCallback(
        save_freq=cfg["checkpoint_freq"],
        save_path=checkpoint_path,
        name_prefix="sac_ckpt",
        verbose=1,
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=eval_path,
        log_path=eval_path,
        eval_freq=cfg["eval_freq"],
        n_eval_episodes=cfg["eval_episodes"],
        deterministic=True,
        render=False,
        verbose=1,
    )

    reward_logger = RewardLoggerCallback()

    return CallbackList([checkpoint_cb, eval_cb, reward_logger]), reward_logger, eval_env


def plot_training_curve(
    timesteps: list[int],
    rewards: list[float],
    run_name: str,
    results_dir: str,
    window: int = 20,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(timesteps, rewards, color="#378ADD", alpha=0.35, linewidth=0.8, label="Episode return")

    if len(rewards) >= window:
        kernel = np.ones(window) / window
        smoothed = np.convolve(rewards, kernel, mode="valid")
        smooth_ts = timesteps[window - 1:]
        ax.plot(smooth_ts, smoothed, color="#378ADD", linewidth=2.0,
                label=f"Smoothed (window={window})")

    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Episodic return")
    ax.set_title(f"SAC Training Curve — HalfCheetah-v5 ({run_name})", fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(results_dir, f"training_curve_{run_name}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved training curve: {path}")


def save_training_summary(
    rewards: list[float],
    elapsed: float,
    run_name: str,
    results_dir: str,
    cfg: dict,
) -> None:
    path = os.path.join(results_dir, f"training_summary_{run_name}.txt")
    final_window = rewards[-100:] if len(rewards) >= 100 else rewards

    with open(path, "w") as f:
        f.write(f"SAC Training Summary — HalfCheetah-v5\n")
        f.write(f"Run: {run_name}\n")
        f.write("=" * 45 + "\n\n")

        f.write("[Hyperparameters]\n")
        for k, v in cfg.items():
            f.write(f"  {k:<20}: {v}\n")

        f.write("\n[Training results]\n")
        f.write(f"  Total episodes      : {len(rewards)}\n")
        f.write(f"  Total wall time (s) : {elapsed:.1f}\n")
        f.write(f"  Final-100 mean      : {np.mean(final_window):.2f}\n")
        f.write(f"  Final-100 std       : {np.std(final_window):.2f}\n")
        f.write(f"  Final-100 max       : {np.max(final_window):.2f}\n")
        f.write(f"  All-time max        : {np.max(rewards):.2f}\n")

    print(f"Saved summary: {path}")

    print("\n[Training Summary]")
    print("=" * 45)
    print(f"  Total episodes      : {len(rewards)}")
    print(f"  Wall time           : {elapsed:.1f}s  ({elapsed/60:.1f} min)")
    print(f"  Final-100 mean      : {np.mean(final_window):.2f}")
    print(f"  Final-100 std       : {np.std(final_window):.2f}")
    print(f"  All-time max return : {np.max(rewards):.2f}")


def train(cfg: dict, run_name: str) -> None:
    os.makedirs(cfg["log_dir"], exist_ok=True)
    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)
    os.makedirs(cfg["results_dir"], exist_ok=True)

    tb_log = os.path.join(cfg["log_dir"], f"sac_halfcheetah_{run_name}")

    print("=" * 55)
    print("  SAC Training — HalfCheetah-v5")
    print("=" * 55)
    print(f"  Run name        : {run_name}")
    print(f"  Total timesteps : {cfg['total_timesteps']:,}")
    print(f"  Seed            : {cfg['seed']}")
    print(f"  TensorBoard dir : {tb_log}")
    print("-" * 55)

    env = make_env(cfg["env_id"], cfg["seed"])

    policy_kwargs = {"net_arch": cfg["net_arch"]}

    model = SAC(
        policy=cfg["policy"],
        env=env,
        learning_rate=cfg["learning_rate"],
        buffer_size=cfg["buffer_size"],
        learning_starts=cfg["learning_starts"],
        batch_size=cfg["batch_size"],
        tau=cfg["tau"],
        gamma=cfg["gamma"],
        train_freq=cfg["train_freq"],
        gradient_steps=cfg["gradient_steps"],
        ent_coef=cfg["ent_coef"],
        target_entropy=cfg["target_entropy"],
        use_sde=cfg["use_sde"],
        policy_kwargs=policy_kwargs,
        tensorboard_log=tb_log,
        seed=cfg["seed"],
        verbose=1,
        device="auto",  # uses CUDA if available
    )

    callbacks, reward_logger, eval_env = build_callbacks(cfg, run_name)

    t0 = time.time()
    model.learn(
        total_timesteps=cfg["total_timesteps"],
        callback=callbacks,
        progress_bar=True,
        reset_num_timesteps=True,
        tb_log_name="SAC",
    )
    elapsed = time.time() - t0

    # Save final model
    final_model_path = os.path.join(
        cfg["checkpoint_dir"], f"sac_{run_name}", "sac_final"
    )
    model.save(final_model_path)
    print(f"\nSaved final model: {final_model_path}.zip")

    env.close()
    eval_env.close()

    rewards = reward_logger.episode_rewards
    timesteps = reward_logger.episode_timesteps

    if rewards:
        plot_training_curve(timesteps, rewards, run_name, cfg["results_dir"])
        save_training_summary(rewards, elapsed, run_name, cfg["results_dir"], cfg)
    else:
        print("Warning: no episodic returns were logged — Monitor wrapper may be missing.")

    print(f"\n[Done] Run '{run_name}' complete.")
    print(f"  View TensorBoard: tensorboard --logdir {cfg['log_dir']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SAC on HalfCheetah-v5")
    parser.add_argument("--timesteps", type=int, default=None,
                        help="Override total training timesteps")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override random seed")
    parser.add_argument("--name", type=str, default=None,
                        help="Run name suffix for logs and checkpoints")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override batch size")
    parser.add_argument("--gamma", type=float, default=None,
                        help="Override discount factor")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = CONFIG.copy()

    if args.timesteps is not None:
        cfg["total_timesteps"] = args.timesteps
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.lr is not None:
        cfg["learning_rate"] = args.lr
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.gamma is not None:
        cfg["gamma"] = args.gamma

    run_name = args.name if args.name else f"seed{cfg['seed']}"
    train(cfg, run_name)
