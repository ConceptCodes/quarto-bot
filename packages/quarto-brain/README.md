# Quarto Brain

Reinforcement Learning and Strategy agents for the Quarto game.

## Training with PufferLib

We use [PufferLib](https://github.com/PufferAI/PufferLib) for high-performance vectorized training.

### Usage

Run the training script from the project root:

```bash
uv run python packages/quarto-brain/quarto_brain/train_puffer.py \
    --num-envs 4 \
    --rollout-steps 1024 \
    --total-timesteps 1000000 \
    --device auto
```

### Key Components

- **`quarto_brain/puffer_env.py`**: Wraps the `QuartoEnv` in PufferLib's emulation layer, flattening the observation space for neural networks.
- **`quarto_brain/train_puffer.py`**: The PPO training loop. Supports multiprocessing and Apple Silicon (MPS) acceleration.

## Agents

- **MCTS Agent**: Monte Carlo Tree Search implementation for classic gameplay.
- **RL Agent**: PPO-trained neural network policy.
