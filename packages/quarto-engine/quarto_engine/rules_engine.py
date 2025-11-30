"""
Quarto Rules Engine

Handles game rules, win conditions, move validation, and game flow for Quarto.
"""

from typing import List, Tuple, Optional, Set
from .game_state import GameState, QuartoPiece, Player, GameStatus


class QuartoRulesEngine:
    """
    Rules engine for Quarto game logic.
    Handles win detection, move validation, and game flow management.
    """

    def __init__(self):
        """Initialize the rules engine."""
        pass

    # ==================== WIN CONDITION DETECTION ====================

    def check_for_win(self, game_state: GameState) -> bool:
        """
        Check if the current board state has a winning condition.

        Args:
            game_state: Current game state

        Returns:
            True if there's a winning line, False otherwise
        """
        # Check all rows
        for row in range(4):
            if self.check_row_win(game_state, row):
                return True

        # Check all columns
        for col in range(4):
            if self.check_column_win(game_state, col):
                return True

        # Check diagonals
        if self.check_diagonal_wins(game_state):
            return True

        return False

    def check_row_win(self, game_state: GameState, row: int) -> bool:
        """
        Check if the specified row has a winning combination.

        Args:
            game_state: Current game state
            row: Row index to check (0-3)

        Returns:
            True if row has 4 pieces with shared attribute
        """
        pieces = [game_state.board[row][col] for col in range(4)]

        # Check if all positions are filled
        if any(piece is None for piece in pieces):
            return False

        return self.check_line_for_shared_attribute(pieces)

    def check_column_win(self, game_state: GameState, col: int) -> bool:
        """
        Check if the specified column has a winning combination.

        Args:
            game_state: Current game state
            col: Column index to check (0-3)

        Returns:
            True if column has 4 pieces with shared attribute
        """
        pieces = [game_state.board[row][col] for row in range(4)]

        # Check if all positions are filled
        if any(piece is None for piece in pieces):
            return False

        return self.check_line_for_shared_attribute(pieces)

    def check_diagonal_wins(self, game_state: GameState) -> bool:
        """
        Check if either diagonal has a winning combination.

        Args:
            game_state: Current game state

        Returns:
            True if either diagonal has 4 pieces with shared attribute
        """
        # Check main diagonal (top-left to bottom-right)
        main_diagonal = [game_state.board[i][i] for i in range(4)]
        if all(piece is not None for piece in main_diagonal):
            if self.check_line_for_shared_attribute(main_diagonal):
                return True

        # Check anti-diagonal (top-right to bottom-left)
        anti_diagonal = [game_state.board[i][3 - i] for i in range(4)]
        if all(piece is not None for piece in anti_diagonal):
            if self.check_line_for_shared_attribute(anti_diagonal):
                return True

        return False

    def check_line_for_shared_attribute(self, pieces: List[QuartoPiece]) -> bool:
        """
        Check if 4 pieces share at least one common attribute.

        Args:
            pieces: List of 4 QuartoPiece objects

        Returns:
            True if all pieces share tall/short, square/round, light/dark, or solid/hollow
        """
        if len(pieces) != 4:
            return False

        attributes = [
            (piece.tall for piece in pieces),
            (piece.square for piece in pieces),
            (piece.light for piece in pieces),
            (piece.solid for piece in pieces),
        ]

        # Check if all pieces share at least one attribute
        return any(len(set(attr)) == 1 for attr in attributes)

    # ==================== MOVE VALIDATION ====================

    def is_valid_placement(
        self, game_state: GameState, row: int, col: int, piece: QuartoPiece
    ) -> bool:
        """
        Validate if a piece can be placed at the specified position.

        Args:
            game_state: Current game state
            row: Target row (0-3)
            col: Target column (0-3)
            piece: Piece to place

        Returns:
            True if placement is valid
        """
        if (
            not (0 <= row < 4 and 0 <= col < 4)
            or game_state.board[row][col] is not None
        ):
            return False

        # Check if the piece matches the selected piece
        if game_state.selected_piece is None:
            return False

        if piece.id != game_state.selected_piece.id:
            return False

        return True

    def is_valid_piece_selection(self, game_state: GameState, piece_id: int) -> bool:
        """
        Validate if a piece can be selected for the opponent.

        Args:
            game_state: Current game state
            piece_id: ID of piece to select (0-15)

        Returns:
            True if piece selection is valid
        """
        if piece_id < 0 or piece_id >= 16:
            return False

        if piece_id not in game_state.available_pieces:
            return False

        if game_state.selected_piece is not None:
            return False

        return True

    def get_valid_placements(self, game_state: GameState) -> List[Tuple[int, int]]:
        """
        Get all valid positions where the current selected piece can be placed.

        Args:
            game_state: Current game state

        Returns:
            List of (row, col) tuples for valid placements
        """
        if game_state.selected_piece is None:
            return []

        valid_positions = []
        for row in range(4):
            for col in range(4):
                if self.is_valid_placement(
                    game_state, row, col, game_state.selected_piece
                ):
                    valid_positions.append((row, col))

        return valid_positions

    def get_available_pieces_for_selection(self, game_state: GameState) -> Set[int]:
        """
        Get all pieces that can be selected for the opponent.

        Args:
            game_state: Current game state

        Returns:
            Set of piece IDs available for selection
        """
        # Return a copy of available pieces
        return game_state.available_pieces.copy()

    # ==================== GAME FLOW MANAGEMENT ====================

    def make_move(self, game_state: GameState, row: int, col: int) -> bool:
        """
        Execute a piece placement move.

        Args:
            game_state: Current game state (modified in-place)
            row: Target row (0-3)
            col: Target column (0-3)

        Returns:
            True if move was successful
        """
        if not self.is_valid_placement(game_state, row, col, game_state.selected_piece):
            return False

        # Place the piece on the board
        game_state.board[row][col] = game_state.selected_piece
        game_state.available_pieces.remove(game_state.selected_piece.id)
        game_state.turn_count += 1

        # Clear the selected piece after placement
        game_state.selected_piece = None

        # Update game status and check for win
        self.update_game_status(game_state)

        return True

    def select_piece_for_opponent(self, game_state: GameState, piece_id: int) -> bool:
        """
        Select a piece for the opponent to place next.

        Args:
            game_state: Current game state (modified in-place)
            piece_id: ID of piece to select (0-15)

        Returns:
            True if selection was successful
        """
        if not self.is_valid_piece_selection(game_state, piece_id):
            return False

        # Create the piece object from ID
        game_state.selected_piece = self.create_piece_from_id(piece_id)

        # Remove the selected piece from available pieces
        game_state.available_pieces.remove(piece_id)

        # Switch to the next player's turn
        self.switch_player(game_state)

        return True

    def switch_player(self, game_state: GameState) -> None:
        """
        Switch to the next player's turn.

        Args:
            game_state: Current game state (modified in-place)
        """
        if game_state.current_player == Player.PLAYER1:
            game_state.current_player = Player.PLAYER2
        elif game_state.current_player == Player.PLAYER2:
            game_state.current_player = Player.PLAYER1
        elif game_state.current_player == Player.ROBOT:
            # If robot is playing, switch back to human player
            game_state.current_player = Player.PLAYER1
        else:
            # Default case
            game_state.current_player = Player.PLAYER1

    def update_game_status(self, game_state: GameState) -> None:
        """
        Update the game status based on current board state.
        Sets winner and game_status fields.

        Args:
            game_state: Current game state (modified in-place)
        """
        # Check for win condition
        if self.check_for_win(game_state):
            game_state.game_status = GameStatus.FINISHED
            game_state.winner = game_state.current_player
        # Check for draw
        elif self.is_draw(game_state):
            game_state.game_status = GameStatus.DRAW
            game_state.winner = None
        else:
            game_state.game_status = GameStatus.ONGOING
            game_state.winner = None

    # ==================== GAME STATE ANALYSIS ====================

    def is_game_over(self, game_state: GameState) -> bool:
        """
        Check if the game has ended (win or draw).

        Args:
            game_state: Current game state

        Returns:
            True if game is over
        """
        return game_state.game_status in [GameStatus.FINISHED, GameStatus.DRAW]

    def is_draw(self, game_state: GameState) -> bool:
        """
        Check if the game is a draw (board full, no winner).

        Args:
            game_state: Current game state

        Returns:
            True if game is a draw
        """
        # Check if board is full (no available pieces) and no winner
        return len(game_state.available_pieces) == 0 and not self.check_for_win(
            game_state
        )

    def get_winning_line(
        self, game_state: GameState
    ) -> Optional[List[Tuple[int, int]]]:
        """
        Get the positions of the winning line, if any.

        Args:
            game_state: Current game state

        Returns:
            List of (row, col) positions forming winning line, or None
        """
        # Check rows
        for row in range(4):
            if self.check_row_win(game_state, row):
                return [(row, col) for col in range(4)]

        # Check columns
        for col in range(4):
            if self.check_column_win(game_state, col):
                return [(row, col) for row in range(4)]

        # Check main diagonal
        main_diagonal = [game_state.board[i][i] for i in range(4)]
        if all(piece is not None for piece in main_diagonal):
            if self.check_line_for_shared_attribute(main_diagonal):
                return [(i, i) for i in range(4)]

        # Check anti-diagonal
        anti_diagonal = [game_state.board[i][3 - i] for i in range(4)]
        if all(piece is not None for piece in anti_diagonal):
            if self.check_line_for_shared_attribute(anti_diagonal):
                return [(i, 3 - i) for i in range(4)]

        return None

    # ==================== UTILITY METHODS ====================

    def create_piece_from_id(self, piece_id: int) -> QuartoPiece:
        """
        Create a QuartoPiece object from its ID (0-15).

        Args:
            piece_id: Unique piece identifier

        Returns:
            QuartoPiece with attributes based on binary representation of ID
        """
        if not (0 <= piece_id <= 15):
            raise ValueError(f"Piece ID must be between 0 and 15, got {piece_id}")

        # Extract attributes from binary representation
        # Bit 0: tall (1) / short (0)
        # Bit 1: square (1) / round (0)
        # Bit 2: light (1) / dark (0)
        # Bit 3: solid (1) / hollow (0)
        tall = bool(piece_id & 1)
        square = bool(piece_id & 2)
        light = bool(piece_id & 4)
        solid = bool(piece_id & 8)

        return QuartoPiece(
            id=piece_id, tall=tall, square=square, light=light, solid=solid
        )

    def get_all_pieces(self) -> List[QuartoPiece]:
        """
        Generate all 16 possible Quarto pieces.

        Returns:
            List of all QuartoPiece objects
        """
        return [self.create_piece_from_id(piece_id) for piece_id in range(16)]

    def reset_game(self, game_state: GameState) -> None:
        """
        Reset the game state to initial conditions.

        Args:
            game_state: Game state to reset (modified in-place)
        """
        # Reset board to empty
        game_state.board = [[None for _ in range(4)] for _ in range(4)]

        # Reset available pieces to all 16 pieces
        game_state.available_pieces = set(range(16))

        # Clear selected piece
        game_state.selected_piece = None

        # Reset game status
        game_state.game_status = GameStatus.ONGOING
        game_state.winner = None

        # Reset turn counter
        game_state.turn_count = 0

        # Reset to first player
        game_state.current_player = Player.PLAYER1
