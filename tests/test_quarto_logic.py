import numpy as np

from quarto_engine.game_state import GameState, Player
from quarto_engine.rules_engine import QuartoRulesEngine
from quarto_brain.env import QuartoEnv


def test_row_win_detection():
    rules = QuartoRulesEngine()
    state = GameState(game_id="t", current_player=Player.PLAYER1)
    # Four tall pieces in top row -> win
    pieces = [
        rules.create_piece_from_id(1),
        rules.create_piece_from_id(3),
        rules.create_piece_from_id(5),
        rules.create_piece_from_id(7),
    ]
    for c, p in enumerate(pieces):
        state.board[0][c] = p
        state.available_pieces.discard(p.id)
    assert rules.check_for_win(state) is True


def test_action_mask_matches_valid_actions():
    env = QuartoEnv(opponent_type=None)
    obs, info = env.reset()
    valid = env.get_valid_actions()
    mask = info["action_mask"]
    assert mask.shape == (32,)
    assert mask.sum() == len(valid)
    assert set(np.nonzero(mask)[0].tolist()) == set(valid)

