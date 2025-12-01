import argparse
import json
import os
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

import pufferlib.vector

from quarto_brain.puffer_env import env_creator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Quarto agent with PufferLib PPO."
    )
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--rollout-steps", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--log-dir", type=str, default="data/quarto_brain")
    parser.add_argument("--checkpoint-dir", type=str, default="models/quarto_brain")
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--opponent",
        type=str,
        default="random",
        choices=["random", "mcts"],
        help="Type of opponent to train against",
    )
    return parser.parse_args()


def select_device(preferred: str) -> torch.device:
    if preferred == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(preferred)


class ActorCritic(nn.Module):
    """MLP Policy for Discrete Action Space."""

    def __init__(
        self,
        obs_shape: Tuple[int, ...],
        action_dim: int,
        hidden_sizes: Tuple[int, ...] = (256, 256),
    ):
        super().__init__()
        obs_dim = int(np.prod(obs_shape))

        # Shared backbone or separate? Let's use separate for simplicity first, or shared.
        # Shared is standard for PPO.
        layers: List[nn.Module] = []
        last_dim = obs_dim
        for size in hidden_sizes:
            layers.append(nn.Linear(last_dim, size))
            layers.append(nn.LayerNorm(size))
            layers.append(nn.Tanh())  # Tanh often better for RL than GELU/ReLU
            last_dim = size

        self.backbone = nn.Sequential(*layers)
        self.actor = nn.Linear(last_dim, action_dim)
        self.critic = nn.Linear(last_dim, 1)

    def forward(self, obs: torch.Tensor, mask: torch.Tensor = None):
        # Flatten observation: (Batch, *ObsShape) -> (Batch, FlatDim)
        flat_obs = obs.view(obs.shape[0], -1)
        hidden = self.backbone(flat_obs)

        # Actor
        logits = self.actor(hidden)
        if mask is not None:
            logits = logits.masked_fill(mask == 0, -1e8)
        dist = Categorical(logits=logits)

        # Critic
        value = self.critic(hidden).squeeze(-1)

        return dist, value

    def get_value(self, obs: torch.Tensor):
        flat_obs = obs.view(obs.shape[0], -1)
        hidden = self.backbone(flat_obs)
        return self.critic(hidden).squeeze(-1)

    def get_action_and_value(self, obs: torch.Tensor, action=None, mask=None):
        dist, value = self.forward(obs, mask)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value


def log_metrics(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(payload) + "\n")


def main():
    args = parse_args()
    device = select_device(args.device)
    print(f"Device: {device}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create Vector Env
    backend = (
        pufferlib.vector.Multiprocessing
        if args.num_envs > 1
        else pufferlib.vector.Serial
    )
    vec_kwargs = {"seed": args.seed, "num_envs": args.num_envs, "backend": backend}

    print(f"Building {args.num_envs} environments...")
    env_fn = env_creator(opponent_type=args.opponent)
    vec_env = pufferlib.vector.make(env_fn, **vec_kwargs)

    obs_shape = vec_env.single_observation_space.shape
    action_dim = vec_env.single_action_space.n
    print(f"Obs shape: {obs_shape}, Action dim: {action_dim}")

    agent = ActorCritic(obs_shape, action_dim).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    global_step = 0
    if args.resume_from:
        print(f"Resuming checkpoint: {args.resume_from}")
        agent.load_state_dict(torch.load(args.resume_from, map_location=device))

        # Try to parse global_step from filename
        # Expected format: "model_step_{step}.pt"
        try:
            filename = Path(args.resume_from).stem
            # model_step_123 -> 123
            step_str = filename.split("_")[-1]
            global_step = int(step_str)
            print(f"Resuming from global_step: {global_step}")
        except ValueError:
            print("Could not parse global_step from filename, starting from 0")

    # Buffers
    obs = torch.zeros((args.rollout_steps, args.num_envs) + obs_shape).to(device)
    actions = torch.zeros((args.rollout_steps, args.num_envs)).to(device)
    logprobs = torch.zeros((args.rollout_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.rollout_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.rollout_steps, args.num_envs)).to(device)
    values = torch.zeros((args.rollout_steps, args.num_envs)).to(device)
    masks = torch.zeros((args.rollout_steps, args.num_envs, action_dim)).to(device)

    start_time = time.time()

    # Initialize environment
    next_obs, next_infos = vec_env.reset()
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    # Initialize mask
    # next_infos is a list of dicts (VectorEnv)
    next_mask = torch.tensor(np.stack([i["action_mask"] for i in next_infos])).to(
        device
    )

    batch_size = args.rollout_steps * args.num_envs
    num_updates = args.total_timesteps // batch_size
    start_update = global_step // batch_size

    metrics_path = Path(args.log_dir) / "metrics.jsonl"
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting training from step {global_step} (Update {start_update})...")
    print(
        f"Target total timesteps: {args.total_timesteps} (Total Updates: {num_updates})"
    )

    for update in range(start_update + 1, num_updates + 1):
        # Annealing the learning rate
        frac = 1.0 - (update - 1.0) / num_updates
        lrnow = frac * args.learning_rate
        optimizer.param_groups[0]["lr"] = lrnow

        # ROLLOUT
        for step in range(args.rollout_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done
            masks[step] = next_mask

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(
                    next_obs, mask=next_mask
                )
                values[step] = value

            actions[step] = action
            logprobs[step] = logprob

            # Execute step
            next_obs_np, reward, term, trunc, infos = vec_env.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs = torch.Tensor(next_obs_np).to(device)
            next_done = torch.Tensor(term | trunc).to(device)  # Logical OR

            # Update mask
            next_mask = torch.tensor(np.stack([i["action_mask"] for i in infos])).to(
                device
            )

            # Log episode returns
            for info in infos:
                if (
                    "episode" in info
                ):  # Gymnasium standard, but Puffer might wrap differently
                    # PufferLib typically handles auto-reset.
                    # We need to rely on PufferLib's info structure if using Emulation.
                    # Standard Gym returns 'final_info' in info list if done.
                    pass

        # Bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)

        # GAE Calculation
        advantages = torch.zeros_like(rewards).to(device)
        lastgaelam = 0
        for t in reversed(range(args.rollout_steps)):
            if t == args.rollout_steps - 1:
                nextnonterminal = 1.0 - next_done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                nextvalues = values[t + 1]
            delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
            advantages[t] = lastgaelam = (
                delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            )
        returns = advantages + values

        # Flatten batch
        b_obs = obs.reshape((-1,) + obs_shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_masks = masks.reshape((-1, action_dim))

        # Optimization
        b_inds = np.arange(args.rollout_steps * args.num_envs)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(
                0, args.rollout_steps * args.num_envs, args.minibatch_size
            ):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions.long()[mb_inds], mask=b_masks[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # Calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                    ]

                mb_advantages = b_advantages[mb_inds]
                if True:  # Normalize advantages?
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                loss = (
                    pg_loss
                    - args.entropy_coef * entropy.mean()
                    + args.value_coef * v_loss
                )

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

        # Logging
        if update % 1 == 0:
            print(
                f"Update {update}/{num_updates} | Step {global_step} | Loss: {loss.item():.3f} | V-Loss: {v_loss.item():.3f}"
            )

        if update % 10 == 0:
            torch.save(
                agent.state_dict(), checkpoint_dir / f"model_step_{global_step}.pt"
            )

    vec_env.close()
    print("Training Complete.")


if __name__ == "__main__":
    main()
