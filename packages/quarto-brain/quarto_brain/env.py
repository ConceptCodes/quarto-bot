import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Optional, Tuple, List

from quarto_engine.rules_engine import QuartoRulesEngine
from quarto_engine.game_state import GameState, Player, GameStatus, QuartoPiece


class QuartoEnv(gym.Env):
    """
    Gymnasium environment for the game of Quarto.

    Action Space: Discrete(32)
      0-15: Place current selected piece at board position (row = action // 4, col = action % 4)
      16-31: Select piece (id = action - 16) for the opponent

    Observation Space: Dict
      'board': Box(low=0, high=1, shape=(4, 4, 5), dtype=int8)
         - 4x4 grid. Channels 0-3 are piece attributes (Tall, Square, Light, Solid).
         - Channel 4 is 1 if piece exists, 0 otherwise.
      'available_pieces': MultiBinary(16)
      'selected_piece': MultiBinary(16) (One-hot encoding of selected piece ID, all 0 if None)
      'current_player': Discrete(2) (0 for Player1, 1 for Player2)
      'phase': Discrete(2) (0 for Placement, 1 for Selection)
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()
        self.rules = QuartoRulesEngine()
        self.game_state = GameState(game_id="gym_quarto", current_player=Player.PLAYER1)
        self.render_mode = render_mode

        # Actions: 0-15 (Place), 16-31 (Select Piece)
        self.action_space = spaces.Discrete(32)

        self.observation_space = spaces.Dict(
            {
                "board": spaces.Box(low=0, high=1, shape=(4, 4, 5), dtype=np.int8),
                "available_pieces": spaces.MultiBinary(16),
                "selected_piece": spaces.MultiBinary(16),
                "current_player": spaces.Discrete(
                    3
                ),  # 0=P1, 1=P2, 2=Robot? Mapped to int
                "phase": spaces.Discrete(2),  # 0=Place, 1=Select
            }
        )

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[dict, dict]:
        super().reset(seed=seed)

        # Reset internal game state
        self.game_state = GameState(game_id="gym_quarto", current_player=Player.PLAYER1)
        self.rules.reset_game(self.game_state)

        # IMPORTANT: Quarto usually starts with P1 selecting a piece for P2.
        # But our state machine starts with selected_piece = None.
        # This implies the first action must be SELECTION.

        return self._get_obs(), self._get_info()

    def step(self, action: int) -> Tuple[dict, float, bool, bool, dict]:
        """
        Execute one step.
        Note: A full "turn" in Quarto involves Placing then Selecting.
        We treat these as separate steps in the environment.
        """
        terminated = False
        truncated = False
        reward = 0.0

        # Determine current phase
        if self.game_state.selected_piece is not None:
            # Phase: Placement (Must place the selected piece)
            if not (0 <= action <= 15):
                # Invalid action type for this phase
                return (
                    self._get_obs(),
                    -10.0,
                    True,
                    False,
                    {"error": "Invalid action: Expected Placement (0-15)"},
                )

            row, col = divmod(action, 4)

            # Check validity
            if not self.rules.is_valid_placement(
                self.game_state, row, col, self.game_state.selected_piece
            ):
                return (
                    self._get_obs(),
                    -10.0,
                    True,
                    False,
                    {"error": "Invalid placement position"},
                )

            # Apply Move
            self.rules.make_move(self.game_state, row, col)

            # Check for Win
            if self.game_state.game_status == GameStatus.FINISHED:
                terminated = True
                reward = 1.0  # Current player won by placing
            elif self.game_state.game_status == GameStatus.DRAW:
                terminated = True
                reward = 0.0

            # Note: After placement, we do NOT switch players yet.
            # We stay in the step loop, waiting for next action (Selection).
            # Unless game is over.

        else:
            # Phase: Selection (Must select a piece for opponent)
            if not (16 <= action <= 31):
                return (
                    self._get_obs(),
                    -10.0,
                    True,
                    False,
                    {"error": "Invalid action: Expected Selection (16-31)"},
                )

            piece_id = action - 16

            # Check validity
            if not self.rules.is_valid_piece_selection(self.game_state, piece_id):
                return (
                    self._get_obs(),
                    -10.0,
                    True,
                    False,
                    {"error": "Invalid piece selection"},
                )

            # Apply Selection
            self.rules.select_piece_for_opponent(self.game_state, piece_id)

            # Now player switches (handled inside select_piece_for_opponent)

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def _get_obs(self) -> dict:
        # 1. Board representation
        board_obs = np.zeros((4, 4, 5), dtype=np.int8)
        for r in range(4):
            for c in range(4):
                p = self.game_state.board[r][c]
                if p is not None:
                    board_obs[r, c, 0] = int(p.tall)
                    board_obs[r, c, 1] = int(p.square)
                    board_obs[r, c, 2] = int(p.light)
                    board_obs[r, c, 3] = int(p.solid)
                    board_obs[r, c, 4] = 1  # Occupied

        # 2. Available pieces
        avail_obs = np.zeros(16, dtype=np.int8)
        for pid in self.game_state.available_pieces:
            avail_obs[pid] = 1

        # 3. Selected piece
        sel_obs = np.zeros(16, dtype=np.int8)
        if self.game_state.selected_piece:
            sel_obs[self.game_state.selected_piece.id] = 1

        # 4. Current Player
        player_map = {Player.PLAYER1: 0, Player.PLAYER2: 1, Player.ROBOT: 1}
        player_id = player_map.get(self.game_state.current_player, 0)

        # 5. Phase
        phase = 0 if self.game_state.selected_piece is not None else 1

        return {
            "board": board_obs,
            "available_pieces": avail_obs,
            "selected_piece": sel_obs,
            "current_player": player_id,
            "phase": phase,
        }

    def _get_info(self) -> dict:
        return {
            "valid_actions": self.get_valid_actions(),
            "turn_count": self.game_state.turn_count,
        }

    def get_valid_actions(self) -> List[int]:
        """Returns a list of valid action IDs for the current state."""
        actions = []
        if self.game_state.selected_piece is not None:
            # Placement phase
            for r in range(4):
                for c in range(4):
                    if self.game_state.board[r][c] is None:
                        actions.append(r * 4 + c)
        else:
            # Selection phase
            for pid in self.game_state.available_pieces:
                actions.append(pid + 16)
        return actions

    def render(self):
        if self.render_mode == "ansi" or self.render_mode == "human":
            print(f"\n--- Turn {self.game_state.turn_count} ---")
            print(f"Player: {self.game_state.current_player}")
            print(
                f"Phase: {'Place Piece' if self.game_state.selected_piece else 'Select Piece'}"
            )
            if self.game_state.selected_piece:
                print(f"Piece to Place: {self.game_state.selected_piece}")

            print("\nBoard:")
            for r in range(4):
                row_str = "|"
                for c in range(4):
                    p = self.game_state.board[r][c]
                    if p:
                        row_str += f"{p}|"
                    else:
                        row_str += "____|"
                print(row_str)

            print("\nAvailable Pieces:")
            print(list(self.game_state.available_pieces))
