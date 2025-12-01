import random
from quarto_engine.rules_engine import QuartoRulesEngine
from quarto_engine.game_state import GameState, Player, GameStatus
from quarto_brain.mcts import QuartoMCTS


def test_mcts_logic():
    print("Testing MCTS vs Random...")
    mcts_wins = 0
    games = 10
    sims = 50

    for i in range(games):
        state = GameState(game_id=str(i), current_player=Player.PLAYER1)
        rules = QuartoRulesEngine()
        mcts = QuartoMCTS(rules, simulations=sims)

        # MCTS is P1
        while state.game_status == GameStatus.ONGOING:
            if state.current_player == Player.PLAYER1:
                # MCTS Turn
                action = mcts.search(state)
                # Apply
                if state.selected_piece is not None:
                    r, c = divmod(action, 4)
                    rules.make_move(state, r, c)
                else:
                    pid = action - 16
                    rules.select_piece_for_opponent(state, pid)
            else:
                # Random Turn
                # Get valid actions
                valid = []
                if state.selected_piece is not None:
                    for r in range(4):
                        for c in range(4):
                            if state.board[r][c] is None:
                                if rules.is_valid_placement(
                                    state, r, c, state.selected_piece
                                ):
                                    valid.append(r * 4 + c)
                else:
                    for pid in state.available_pieces:
                        if rules.is_valid_piece_selection(state, pid):
                            valid.append(pid + 16)

                if not valid:
                    break
                action = random.choice(valid)
                if state.selected_piece is not None:
                    r, c = divmod(action, 4)
                    rules.make_move(state, r, c)
                else:
                    pid = action - 16
                    rules.select_piece_for_opponent(state, pid)

        if state.winner == Player.PLAYER1:
            mcts_wins += 1
            # print(f"Game {i}: MCTS Won")
        else:
            print(f"Game {i}: Random Won (or Draw)")

    print(f"MCTS Win Rate: {mcts_wins}/{games}")


if __name__ == "__main__":
    test_mcts_logic()
