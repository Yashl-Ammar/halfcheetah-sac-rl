"""
verify_env.py
-------------
Phase 2 — Step 1: Verify the HalfCheetah-v5 environment is correctly
installed and behaves as expected before any training begins.

Run:
    python verify_env.py
"""

import gymnasium as gym
import numpy as np


def verify_environment(env_id: str = "HalfCheetah-v5", n_steps: int = 200) -> None:
    print("=" * 55)
    print(f"  Environment Verification: {env_id}")
    print("=" * 55)

    # --- Create environment ---
    env = gym.make(env_id, render_mode=None)

    # --- Observation space ---
    obs_space = env.observation_space
    print(f"\n[Observation space]")
    print(f"  Shape : {obs_space.shape}")
    print(f"  Low   : {obs_space.low[:4]} ...")
    print(f"  High  : {obs_space.high[:4]} ...")

    # --- Action space ---
    act_space = env.action_space
    print(f"\n[Action space]")
    print(f"  Shape : {act_space.shape}")
    print(f"  Low   : {act_space.low}")
    print(f"  High  : {act_space.high}")

    # --- Reset ---
    obs, info = env.reset(seed=42)
    print(f"\n[Reset]")
    print(f"  Initial obs shape : {obs.shape}")
    print(f"  Initial obs sample: {np.round(obs[:4], 4)} ...")
    print(f"  Info keys         : {list(info.keys())}")

    # --- Step loop ---
    print(f"\n[Stepping {n_steps} random actions]")
    total_reward = 0.0
    terminated = truncated = False

    for i in range(n_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            obs, info = env.reset()
            print(f"  Episode ended at step {i + 1} — resetting.")
            break

    print(f"  Cumulative reward over {n_steps} steps: {total_reward:.2f}")

    # --- Reward components (v5 exposes x_velocity and forward_reward) ---
    print(f"\n[Reward info (last step)]")
    for k, v in info.items():
        print(f"  {k}: {v}")

    env.close()

    print("\n[Result] Environment verified successfully.\n")


if __name__ == "__main__":
    verify_environment()
