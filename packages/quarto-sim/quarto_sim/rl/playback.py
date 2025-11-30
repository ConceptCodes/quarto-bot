import argparse
import time
import torch
import numpy as np
from pathlib import Path

from quarto_sim.rl.puffer_config import load_pick_training_config
from quarto_sim.rl.puffer_pick_env import make_puffer_pick_env
from quarto_sim.rl.train_pick_puffer import ActorCritic


def parse_args():
    parser = argparse.ArgumentParser(description="Playback trained policy")
    parser.add_argument("--config", type=str, default="src/config/puffer_pick.yaml")
    parser.add_argument(
        "--checkpoint", type=str, default="models/puffer_pick/policy_latest.pt"
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample from policy instead of using mean",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    train_cfg = load_pick_training_config(args.config)

    # Create single environment with rendering
    print(f"Creating environment with config: {args.config}")
    env = make_puffer_pick_env(config=train_cfg.env, render_mode="human")

    # Setup device
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Initialize model
    obs_shape = env.observation_space.shape
    if obs_shape is None:
        raise ValueError("Observation space shape is None")

    action_shape = env.action_space.shape
    if action_shape is None:
        raise ValueError("Action space shape is None")

    action_dim = int(np.prod(action_shape))
    print(f"Observation shape: {obs_shape}, Action dim: {action_dim}")

    policy = ActorCritic(obs_shape, action_dim).to(device)

    # Load checkpoint
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"Checkpoint not found at {ckpt_path}")
        print("Please run training first: uv run python src/rl/train_pick_puffer.py")
        return

    print(f"Loading checkpoint from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    policy.load_state_dict(checkpoint["state_dict"])
    policy.eval()

    # Playback loop
    print("Starting playback... Press Ctrl+C to stop.")

    try:
        while True:
            obs, _ = env.reset()
            done = False
            truncated = False
            episode_reward = 0
            step = 0
            info = {}

            while not (done or truncated):
                # Prepare observation
                obs_tensor = torch.as_tensor(
                    obs, dtype=torch.float32, device=device
                ).unsqueeze(0)

                # Get action
                with torch.no_grad():
                    dist, _ = policy(obs_tensor)
                    if args.stochastic:
                        action = dist.sample()
                    else:
                        action = dist.mean

                action_np = action.cpu().numpy().flatten()

                # Step environment
                obs, reward, done, truncated, info = env.step(action_np)
                episode_reward += reward
                step += 1

                # Render
                env.render()

                # Sync with 60Hz
                time.sleep(1 / 60.0)

            success = info.get("episode_success", False)
            status = "SUCCESS" if success else "FAILED"
            print(
                f"Episode finished. Reward: {episode_reward:.2f}, Steps: {step}, Status: {status}"
            )
            time.sleep(1.0)  # Pause between episodes

    except KeyboardInterrupt:
        print("\nPlayback stopped by user")
    finally:
        env.close()


if __name__ == "__main__":
    main()
