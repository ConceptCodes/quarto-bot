import time
from quarto_engine.game_state import Player, GameStatus
from .env import QuartoEnv
from .rl_agent import RandomAgent, MCTSAgent


def train():
    env = QuartoEnv(render_mode="ansi")

    # Initialize agents
    # Reduced simulations for quick verification
    agent1 = MCTSAgent(simulations=50, time_limit=0.5)
    agent2 = RandomAgent()

    episodes = 1
    wins = {Player.PLAYER1: 0, Player.PLAYER2: 0, "Draw": 0}

    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        print(f"\n=== Episode {ep + 1} ===")

        while not done:
            # Determine current player based on env state logic
            # Env state tracks current_player correctly
            current_player = env.game_state.current_player

            if current_player == Player.PLAYER1:
                action = agent1.get_action(env.game_state)
            else:
                action = agent2.get_action(env.game_state)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if done:
                env.render()
                if env.game_state.game_status == GameStatus.FINISHED:
                    winner = env.game_state.winner
                    wins[winner] += 1
                    print(f"Winner: {winner}")
                else:
                    wins["Draw"] += 1
                    print("Draw!")

    print("\nResults:")
    print(wins)


if __name__ == "__main__":
    train()
