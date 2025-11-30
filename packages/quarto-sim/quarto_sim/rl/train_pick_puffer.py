from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch import nn

import pufferlib.vector

from quarto_sim.rl.puffer_config import (
    PickAndPlaceTrainingConfig,
    load_pick_training_config,
)
from quarto_sim.rl.puffer_pick_env import env_creator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train pick-and-place policy with PufferLib PPO."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="packages/quarto-sim/config/puffer_pick.yaml",
        help="Path to the YAML config that defines env + training defaults.",
    )
    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=1e-3)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--target-kl", type=float, default=None)
    parser.add_argument(
        "--print-every",
        type=int,
        default=1,
        help="How often (in updates) to print metrics.",
    )
    parser.add_argument(
        "--checkpoint-frequency",
        type=int,
        default=10,
        help="Save a checkpoint every N updates (0 disables periodic checkpoints).",
    )
    return parser.parse_args()


def select_device(preferred: str) -> torch.device:
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if preferred == "mps" and torch.backends.mps.is_available():  # type: ignore[attr-defined]
        return torch.device("mps")
    if preferred == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():  # type: ignore[attr-defined]
        return torch.device("mps")
    return torch.device("cpu")


class ActorCritic(nn.Module):
    """Simple MLP policy/value network for continuous control."""

    def __init__(
        self,
        obs_shape: Tuple[int, ...],
        action_dim: int,
        hidden_sizes: Tuple[int, ...] = (256, 256),
    ):
        super().__init__()
        obs_dim = int(np.prod(obs_shape))
        layers: List[nn.Module] = []
        last_dim = obs_dim
        for size in hidden_sizes:
            layers.append(nn.Linear(last_dim, size))
            layers.append(nn.LayerNorm(size))
            layers.append(nn.GELU())
            last_dim = size

        self.backbone = nn.Sequential(*layers)
        self.policy_head = nn.Linear(last_dim, action_dim)
        self.value_head = nn.Linear(last_dim, 1)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def _forward_backbone(self, obs: torch.Tensor) -> torch.Tensor:
        flat = obs.view(obs.shape[0], -1)
        return self.backbone(flat)

    def forward(self, obs: torch.Tensor):
        hidden = self._forward_backbone(obs)
        mean = self.policy_head(hidden)
        std = torch.exp(self.log_std).expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        value = self.value_head(hidden).squeeze(-1)
        return dist, value


def build_vector_env(cfg: PickAndPlaceTrainingConfig):
    """Create a vectorized environment stack using PufferLib."""
    backend = (
        pufferlib.vector.Multiprocessing
        if cfg.num_envs > 1
        else pufferlib.vector.Serial
    )
    vec_kwargs = {"seed": cfg.seed, "num_envs": cfg.num_envs, "backend": backend}
    if backend is pufferlib.vector.Multiprocessing:
        cpu_count = os.cpu_count() or 1
        num_workers = min(cfg.num_envs, cpu_count)
        while cfg.num_envs % num_workers != 0 and num_workers > 1:
            num_workers -= 1
        vec_kwargs["num_workers"] = max(1, num_workers)
        vec_kwargs["batch_size"] = cfg.num_envs
    env_fn = env_creator(cfg.env)
    return pufferlib.vector.make(env_fn, **vec_kwargs)


def log_metrics(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(payload) + "\n")


def main() -> None:
    print("Parsing args...")
    args = parse_args()
    print("Loading config...")
    train_cfg = load_pick_training_config(args.config)
    print(f"Config loaded. Device: {train_cfg.device}, Envs: {train_cfg.num_envs}")
    device = select_device(train_cfg.device)
    print(f"Selected device: {device}")
    torch.manual_seed(train_cfg.seed)
    np.random.seed(train_cfg.seed)

    print("Building vector env...")
    vec_env = build_vector_env(train_cfg)
    print("Vector env built.")
    obs_shape = vec_env.single_observation_space.shape
    action_dim = int(np.prod(vec_env.single_action_space.shape))
    print(f"Obs shape: {obs_shape}, Action dim: {action_dim}")

    policy = ActorCritic(obs_shape, action_dim).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate)

    rollout_steps = train_cfg.rollout_steps
    num_envs = train_cfg.num_envs
    batch_size = rollout_steps * num_envs
    total_updates = max(1, args.total_timesteps // batch_size)

    obs_buf = torch.zeros((rollout_steps, num_envs) + obs_shape, device=device)
    action_buf = torch.zeros((rollout_steps, num_envs, action_dim), device=device)
    logprob_buf = torch.zeros((rollout_steps, num_envs), device=device)
    reward_buf = torch.zeros((rollout_steps, num_envs), device=device)
    done_buf = torch.zeros((rollout_steps, num_envs), device=device)
    value_buf = torch.zeros((rollout_steps, num_envs), device=device)

    metrics_path = Path(train_cfg.log_dir) / "metrics.jsonl"
    checkpoint_dir = Path(train_cfg.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    obs, _ = vec_env.reset(seed=train_cfg.seed)
    next_obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
    global_step = 0
    start_time = time.time()

    for update in range(total_updates):
        episode_returns: List[float] = []
        episode_lengths: List[int] = []
        successes: List[bool] = []

        for step in range(rollout_steps):
            obs_buf[step] = next_obs
            with torch.no_grad():
                dist, value = policy(next_obs)
                action = dist.sample()
                logprob = dist.log_prob(action).sum(-1)

            action_buf[step] = action
            logprob_buf[step] = logprob
            value_buf[step] = value

            actions_np = action.detach().cpu().numpy()
            next_obs_np, rewards, terminals, truncations, infos = vec_env.step(
                actions_np
            )
            reward_buf[step] = torch.as_tensor(
                rewards, dtype=torch.float32, device=device
            )
            done_buf[step] = torch.as_tensor(
                np.logical_or(terminals, truncations),
                dtype=torch.float32,
                device=device,
            )

            next_obs = torch.as_tensor(next_obs_np, dtype=torch.float32, device=device)
            global_step += num_envs

            for info in infos or []:
                if not info:
                    continue
                if "episode_return" in info:
                    episode_returns.append(float(info["episode_return"]))
                if "episode_length" in info:
                    episode_lengths.append(int(info["episode_length"]))
                if "episode_success" in info:
                    successes.append(bool(info["episode_success"]))

        with torch.no_grad():
            next_value = policy(next_obs)[1]

        advantages = torch.zeros_like(reward_buf, device=device)
        lastgaelam = torch.zeros(num_envs, device=device)
        for t in reversed(range(rollout_steps)):
            if t == rollout_steps - 1:
                next_non_terminal = 1.0 - done_buf[t]
                next_values = next_value
            else:
                next_non_terminal = 1.0 - done_buf[t + 1]
                next_values = value_buf[t + 1]

            delta = (
                reward_buf[t]
                + args.gamma * next_values * next_non_terminal
                - value_buf[t]
            )
            lastgaelam = (
                delta + args.gamma * args.gae_lambda * next_non_terminal * lastgaelam
            )
            advantages[t] = lastgaelam

        returns = advantages + value_buf

        b_obs = obs_buf.reshape(batch_size, *obs_shape)
        b_actions = action_buf.reshape(batch_size, action_dim)
        b_logprobs = logprob_buf.reshape(batch_size)
        b_advantages = advantages.reshape(batch_size)
        b_returns = returns.reshape(batch_size)
        b_values = value_buf.reshape(batch_size)

        if torch.std(b_advantages) > 1e-6:
            b_advantages = (b_advantages - b_advantages.mean()) / (
                b_advantages.std() + 1e-8
            )

        inds = np.arange(batch_size)
        clipfracs: List[float] = []
        approx_kls: List[float] = []

        for _ in range(train_cfg.update_epochs):
            np.random.shuffle(inds)
            for start in range(0, batch_size, train_cfg.minibatch_size):
                end = start + train_cfg.minibatch_size
                mb_inds = inds[start:end]
                if len(mb_inds) == 0:
                    continue

                dist, value = policy(b_obs[mb_inds])
                new_logprob = dist.log_prob(b_actions[mb_inds]).sum(-1)
                entropy = dist.entropy().sum(-1).mean()

                ratio = (new_logprob - b_logprobs[mb_inds]).exp()
                approx_kl = torch.mean(b_logprobs[mb_inds] - new_logprob).item()
                approx_kls.append(max(approx_kl, 0.0))

                with torch.no_grad():
                    clipfracs.append(
                        torch.mean(
                            (torch.abs(ratio - 1.0) > args.clip_coef).float()
                        ).item()
                    )

                adv = b_advantages[mb_inds]
                pg_loss = torch.mean(
                    torch.max(
                        -adv * ratio,
                        -adv
                        * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef),
                    )
                )

                value_clipped = b_values[mb_inds] + (value - b_values[mb_inds]).clamp(
                    -args.clip_coef, args.clip_coef
                )
                value_losses = (value - b_returns[mb_inds]) ** 2
                value_losses_clipped = (value_clipped - b_returns[mb_inds]) ** 2
                value_loss = 0.5 * torch.mean(
                    torch.max(value_losses, value_losses_clipped)
                )

                loss = (
                    pg_loss + args.value_coef * value_loss - args.entropy_coef * entropy
                )
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and np.mean(approx_kls) > args.target_kl:
                break

        fps = global_step / max(1e-6, time.time() - start_time)
        metrics = {
            "update": update,
            "global_step": global_step,
            "fps": fps,
            "mean_return": float(np.mean(episode_returns)) if episode_returns else None,
            "mean_length": float(np.mean(episode_lengths)) if episode_lengths else None,
            "success_rate": float(np.mean(successes)) if successes else None,
            "approx_kl": float(np.mean(approx_kls)) if approx_kls else None,
            "clipfrac": float(np.mean(clipfracs)) if clipfracs else None,
        }
        log_metrics(metrics_path, metrics)

        if args.print_every and (update + 1) % args.print_every == 0:
            pretty = {k: v for k, v in metrics.items() if v is not None}
            print(json.dumps(pretty))

        if args.checkpoint_frequency and (update + 1) % args.checkpoint_frequency == 0:
            ckpt_path = checkpoint_dir / f"policy_step_{global_step}.pt"
            torch.save(
                {
                    "state_dict": policy.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "global_step": global_step,
                    "config": train_cfg.to_dict(),
                },
                ckpt_path,
            )

    vec_env.close()
    final_ckpt = checkpoint_dir / "policy_latest.pt"
    torch.save(
        {
            "state_dict": policy.state_dict(),
            "optimizer": optimizer.state_dict(),
            "global_step": global_step,
            "config": train_cfg.to_dict(),
        },
        final_ckpt,
    )
    print(f"Training finished. Final checkpoint saved to {final_ckpt}")


if __name__ == "__main__":
    main()
