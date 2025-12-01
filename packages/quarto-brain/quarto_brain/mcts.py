import math
import random
import copy
import time
from typing import List, Optional, Tuple, Dict
from quarto_engine.game_state import GameState, Player, GameStatus
from quarto_engine.rules_engine import QuartoRulesEngine


class MCTSNode:
    def __init__(self, state: GameState, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action  # Action taken to reach this node
        self.children: List[MCTSNode] = []
        self.visits = 0
        self.value = 0.0
        self.untried_actions = self._get_legal_actions()

    def _get_legal_actions(self) -> List[int]:
        rules = QuartoRulesEngine()
        actions = []
        if self.state.selected_piece is not None:
            # Placement phase
            for r in range(4):
                for c in range(4):
                    if self.state.board[r][c] is None:
                        if rules.is_valid_placement(
                            self.state, r, c, self.state.selected_piece
                        ):
                            actions.append(r * 4 + c)
        else:
            # Selection phase
            for pid in self.state.available_pieces:
                if rules.is_valid_piece_selection(self.state, pid):
                    actions.append(pid + 16)
        return actions

    def is_terminal(self) -> bool:
        return self.state.game_status != GameStatus.ONGOING

    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0

    def best_child(self, c_param=1.414) -> "MCTSNode":
        choices_weights = [
            (child.value / child.visits)
            + c_param * math.sqrt((2 * math.log(self.visits) / child.visits))
            for child in self.children
        ]
        return self.children[choices_weights.index(max(choices_weights))]


class QuartoMCTS:
    def __init__(
        self,
        rules_engine: QuartoRulesEngine,
        simulations: int = 1000,
        time_limit: float = None,
    ):
        self.rules = rules_engine
        self.simulations = simulations
        self.time_limit = time_limit  # seconds

    def search(self, root_state: GameState) -> int:
        root = MCTSNode(state=root_state.model_copy(deep=True))

        start_time = time.time()
        for i in range(self.simulations):
            if self.time_limit and (time.time() - start_time) > self.time_limit:
                break

            node = self._select(root)
            reward = self._simulate(node.state)
            self._backpropagate(node, reward)

        if not root.children:
            # Should not happen unless no moves available
            return random.choice(root._get_legal_actions())

        return root.best_child(c_param=0.0).action

    def _select(self, node: MCTSNode) -> MCTSNode:
        while not node.is_terminal():
            if not node.is_fully_expanded():
                return self._expand(node)
            else:
                node = node.best_child()
        return node

    def _expand(self, node: MCTSNode) -> MCTSNode:
        action = node.untried_actions.pop()
        next_state = node.state.model_copy(deep=True)
        self._apply_action(next_state, action)
        child_node = MCTSNode(state=next_state, parent=node, action=action)
        node.children.append(child_node)
        return child_node

    def _simulate(self, state: GameState) -> float:
        current_state = state.model_copy(deep=True)
        # We need to track who initiated the simulation to determine reward perspective
        # But MCTS values are typically for the node's player?
        # Actually, standard MCTS:
        # If State S is P1's turn -> Child C is P2's turn (or P1's same turn phase).
        # We need to simulate until end.

        while current_state.game_status == GameStatus.ONGOING:
            legal = self._get_legal_actions_sim(current_state)
            if not legal:
                break
            action = random.choice(legal)
            self._apply_action(current_state, action)

        # Reward: +1 if WE won, -1 if opponent won?
        # Typically MCTS nodes store value for the player who Just moved to get there?
        # Or always from Root player perspective?
        # Let's use Root player perspective.

        # Who is the "Root" player?
        # We don't track it here explicitly, but backprop handles alternating turns if standard game.
        # But Quarto has 2-step turns.
        # Let's just return result relative to Player 1 for now, and handle logic in backprop?
        # Simpler: Return dictionary of {Player: Score}

        if current_state.game_status == GameStatus.DRAW:
            return 0.0
        elif current_state.game_status == GameStatus.FINISHED:
            # Winner is current_state.winner
            # We return score for the player who's turn it was at the NODE?
            return (
                1.0 if current_state.winner == Player.PLAYER1 else -1.0
            )  # This is naive

        return 0.0

    def _backpropagate(self, node: MCTSNode, result: float):
        # result is +1.0 for Player 1 win, -1.0 for Player 2 win, 0.0 for Draw
        while node is not None:
            node.visits += 1

            if node.parent is not None:
                # Determine if this outcome was good for the player who made the move (parent)
                parent_player = node.parent.state.current_player
                if parent_player == Player.PLAYER1:
                    # If P1 made the move, positive result (P1 win) is good
                    node.value += result
                else:
                    # If P2 made the move, negative result (P2 win) is good
                    node.value -= result
            else:
                # Root node just accumulates raw result for stats
                node.value += result

            node = node.parent

    def _apply_action(self, state: GameState, action: int):
        if state.selected_piece is not None:
            # Place
            row, col = divmod(action, 4)
            self.rules.make_move(state, row, col)
        else:
            # Select
            pid = action - 16
            self.rules.select_piece_for_opponent(state, pid)

    def _get_legal_actions_sim(self, state: GameState) -> List[int]:
        # Fast version for simulation
        actions = []
        if state.selected_piece is not None:
            for r in range(4):
                for c in range(4):
                    if state.board[r][c] is None:
                        # Assume validity for speed if relying on engine
                        actions.append(r * 4 + c)
        else:
            for pid in state.available_pieces:
                actions.append(pid + 16)
        return actions
