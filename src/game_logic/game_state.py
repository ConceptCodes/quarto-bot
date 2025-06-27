from pydantic import BaseModel, Field
from typing import Optional, List, Set
from enum import Enum


class GameStatus(str, Enum):
    """Game status enumeration."""

    ONGOING = "ongoing"
    FINISHED = "finished"
    DRAW = "draw"


class Player(str, Enum):
    """Player enumeration."""

    PLAYER1 = "player1"
    PLAYER2 = "player2"
    ROBOT = "robot"


class QuartoPiece(BaseModel):
    """
    Represents a Quarto piece with its four binary attributes.
    """

    id: int = Field(..., description="Unique piece ID (0-15)")
    tall: bool = Field(..., description="True if tall, False if short")
    square: bool = Field(..., description="True if square, False if round")
    light: bool = Field(..., description="True if light, False if dark")
    solid: bool = Field(..., description="True if solid, False if hollow")

    def __str__(self) -> str:
        """String representation of the piece."""
        height = "T" if self.tall else "S"
        shape = "Q" if self.square else "R"
        color = "L" if self.light else "D"
        fill = "S" if self.solid else "H"
        return f"{height}:{shape}:{color}:{fill}"


class GameState(BaseModel):
    """
    Represents the complete state of a Quarto game.
    """

    game_id: str = Field(..., description="Unique identifier for the game")
    current_player: Player = Field(
        ..., description="Player whose turn it is to place a piece"
    )
    board: List[List[Optional[QuartoPiece]]] = Field(
        default_factory=lambda: [[None for _ in range(4)] for _ in range(4)],
        description="4x4 game board with pieces",
    )
    available_pieces: Set[int] = Field(
        default_factory=lambda: set(range(16)),
        description="Set of piece IDs that are still available to be placed",
    )
    selected_piece: Optional[QuartoPiece] = Field(
        None,
        description="Piece selected by the previous player for the current player to place",
    )
    game_status: GameStatus = Field(
        default=GameStatus.ONGOING, description="Current status of the game"
    )
    winner: Optional[Player] = Field(
        None, description="Player who has won the game, if applicable"
    )
    turn_count: int = Field(default=0, description="Number of turns played")

    class Config:
        schema_extra = {
            "example": {
                "game_id": "quarto_game_001",
                "current_player": "player1",
                "board": [[None, None, None, None] for _ in range(4)],
                "available_pieces": list(range(16)),
                "selected_piece": None,
                "game_status": "ongoing",
                "winner": None,
                "turn_count": 0,
            }
        }

    def get_piece_at(self, row: int, col: int) -> Optional[QuartoPiece]:
        """Get the piece at the specified board position."""
        if 0 <= row < 4 and 0 <= col < 4:
            return self.board[row][col]
        return None

    def place_piece(self, row: int, col: int, piece: QuartoPiece) -> bool:
        """
        Place a piece on the board at the specified position.
        Returns True if successful, False if position is occupied.
        """
        if (
            0 <= row < 4
            and 0 <= col < 4
            and self.board[row][col] is None
            and piece.id in self.available_pieces
        ):

            self.board[row][col] = piece
            self.available_pieces.remove(piece.id)
            self.selected_piece = None
            self.turn_count += 1
            return True
        return False

    def is_board_full(self) -> bool:
        """Check if the board is completely filled."""
        return len(self.available_pieces) == 0

    def get_empty_positions(self) -> List[tuple]:
        """Get all empty positions on the board."""
        empty_positions = []
        for row in range(4):
            for col in range(4):
                if self.board[row][col] is None:
                    empty_positions.append((row, col))
        return empty_positions
