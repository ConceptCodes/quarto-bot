import argparse
import random
import torch
import numpy as np
import time
from pathlib import Path
from tqdm import tqdm

from quarto_brain.train_puffer import ActorCritic, select_device
from quarto_brain.puffer_env import make_quarto_env
from quarto_brain.mcts import QuartoMCTS
from quarto_engine.game_state import Player, GameStatus
from quarto_engine.rules_engine import QuartoRulesEngine


class RandomAgent:
    def __init__(self, seed=None):
        self.rng = random.Random(seed)

    def get_action(self, game_state):
        # We need valid actions.
        # The QuartoRulesEngine/GameState doesn't have a get_valid_actions convenience on State itself easily accessible without Rules logic?
        # Actually QuartoEnv has it.
        # We can re-implement or use the one from env if we had access.
        # But here we pass game_state.
        # Let's use the helper helper function in MCTS or just implement it briefly.
        rules = QuartoRulesEngine()
        actions = []
        if game_state.selected_piece is not None:
            # Placement
            for r in range(4):
                for c in range(4):
                    if game_state.board[r][c] is None:
                        if rules.is_valid_placement(
                            game_state, r, c, game_state.selected_piece
                        ):
                            actions.append(r * 4 + c)
        else:
            # Selection
            for pid in game_state.available_pieces:
                if rules.is_valid_piece_selection(game_state, pid):
                    actions.append(pid + 16)

        return self.rng.choice(actions) if actions else 0


class MCTSAgent:
    def __init__(self, simulations=100):
        self.mcts = QuartoMCTS(QuartoRulesEngine(), simulations=simulations)

    def get_action(self, game_state):
        return self.mcts.search(game_state)


def load_model(path, obs_shape, action_dim, device):
    model = ActorCritic(obs_shape, action_dim).to(device)
    # Load checkpoint
    if not Path(path).exists():
        raise FileNotFoundError(f"Model not found at {path}")

    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def run_evaluation(
    model_path, opponent_type="random", num_games=100, mcts_sims=50, device="auto"
):
    device = select_device(device)
    print(f"Evaluating {model_path} against {opponent_type} on {device}")

    # Setup Environment (Puffer Wrapper)
    env = make_quarto_env()
    obs_shape = env.observation_space.shape
    action_dim = env.action_space.n

    # Load Model
    agent = load_model(model_path, obs_shape, action_dim, device)

    # Setup Opponent
    if opponent_type == "random":
        opponent = RandomAgent()
    elif opponent_type == "mcts":
        opponent = MCTSAgent(simulations=mcts_sims)
    else:
        raise ValueError(f"Unknown opponent: {opponent_type}")

    # Metrics
    wins_as_p1 = 0
    wins_as_p2 = 0
    draws = 0
    losses_as_p1 = 0
    losses_as_p2 = 0

    # Run Games
    # We alternate who is P1.
    # Games 0..N/2: PPO is P1
    # Games N/2..N: PPO is P2

    pbar = tqdm(range(num_games))
    for i in pbar:
        obs, info = env.reset()
        done = False

        # Determine PPO side
        ppo_player = Player.PLAYER1 if i < num_games // 2 else Player.PLAYER2

        # PufferEnv flattens obs. Accessing internal game state:
        # env.env is QuartoEnv
        internal_env = env.env

        while not done:
            current_player = internal_env.game_state.current_player

            if current_player == ppo_player:
                # PPO Turn
                obs_tensor = torch.Tensor(obs).unsqueeze(0).to(device)
                mask_tensor = torch.Tensor(info["action_mask"]).unsqueeze(0).to(device)
                with torch.no_grad():
                    action_tensor, _, _, _ = agent.get_action_and_value(
                        obs_tensor, mask=mask_tensor
                    )
                    action = action_tensor.item()
            else:
                # Opponent Turn
                action = opponent.get_action(internal_env.game_state)

            obs, reward, term, trunc, info = env.step(action)
            done = term or trunc

            if done:
                # Determine Winner
                status = internal_env.game_state.game_status
                winner = None

                if status == GameStatus.FINISHED:
                    winner = internal_env.game_state.winner
                elif status == GameStatus.DRAW:
                    draws += 1
                elif status == GameStatus.ONGOING:
                    # Disqualification due to invalid move
                    # The player who acted (current_player) loses
                    winner = (
                        Player.PLAYER2
                        if current_player == Player.PLAYER1
                        else Player.PLAYER1
                    )

                if winner is not None:
                    if winner == ppo_player:
                        if ppo_player == Player.PLAYER1:
                            wins_as_p1 += 1
                        else:
                            wins_as_p2 += 1
                    else:
                        if ppo_player == Player.PLAYER1:
                            losses_as_p1 += 1
                        else:
                            losses_as_p2 += 1

        # Update progress bar description
        p1_games = (i + 1) if i < num_games // 2 else num_games // 2
        p2_games = (i + 1 - num_games // 2) if i >= num_games // 2 else 0

        p1_winrate = wins_as_p1 / p1_games if p1_games > 0 else 0
        p2_winrate = wins_as_p2 / p2_games if p2_games > 0 else 0

        pbar.set_description(f"P1 WR: {p1_winrate:.2f} | P2 WR: {p2_winrate:.2f}")

    print("\n--- Evaluation Results ---")
    print(f"Total Games: {num_games}")
    print(f"PPO as P1: {wins_as_p1} Wins, {losses_as_p1} Losses")
    print(f"PPO as P2: {wins_as_p2} Wins, {losses_as_p2} Losses")
    print(f"Draws: {draws}")
    print(f"Overall Winrate: {(wins_as_p1 + wins_as_p2) / num_games:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str, help="Path to .pt model file")
    parser.add_argument(
        "--opponent", type=str, default="random", choices=["random", "mcts"]
    )
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--mcts-sims", type=int, default=50)
    parser.add_argument("--device", type=str, default="auto")

    args = parser.parse_args()

    run_evaluation(
        args.model_path,
        opponent_type=args.opponent,
        num_games=args.games,
        mcts_sims=args.mcts_sims,
        device=args.device,
    )
