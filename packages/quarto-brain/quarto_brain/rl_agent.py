from typing import Optional
import random
from quarto_engine.game_state import GameState, Player
from quarto_engine.rules_engine import QuartoRulesEngine
from .mcts import QuartoMCTS


class QuartoAgent:
    """
    Base class for Quarto Agents.
    """

    def get_action(self, state: GameState) -> int:
        raise NotImplementedError


class RandomAgent(QuartoAgent):
    """
    Agent that takes random valid actions.
    """

    def __init__(self):
        self.rules = QuartoRulesEngine()

    def get_action(self, state: GameState) -> int:
        valid_actions = []
        if state.selected_piece is not None:
            # Placement phase
            for r in range(4):
                for c in range(4):
                    if self.rules.is_valid_placement(state, r, c, state.selected_piece):
                        valid_actions.append(r * 4 + c)
        else:
            # Selection phase
            for pid in state.available_pieces:
                if self.rules.is_valid_piece_selection(state, pid):
                    valid_actions.append(pid + 16)

        if not valid_actions:
            return 0  # Should not happen
        return random.choice(valid_actions)


class MCTSAgent(QuartoAgent):
    """
    Agent using Monte Carlo Tree Search.
    """

    def __init__(self, simulations=1000, time_limit=1.0):
        self.rules = QuartoRulesEngine()
        self.mcts = QuartoMCTS(
            self.rules, simulations=simulations, time_limit=time_limit
        )

    def get_action(self, state: GameState) -> int:
        return self.mcts.search(state)
