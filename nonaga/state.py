from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .geometry import CORNER_COORDS, Coord, hex_radius_two
from .symmetry import apply_transform, normalize_translation

Color = Literal[1, -1]
RED: Color = 1
BLACK: Color = -1


def other_color(color: Color) -> Color:
    return BLACK if color == RED else RED


@dataclass(frozen=True, slots=True)
class GameState:
    discs: frozenset[Coord]
    red: tuple[Coord, Coord, Coord]
    black: tuple[Coord, Coord, Coord]
    side_to_move: Color = RED
    forbidden_disc: Coord | None = None
    ply: int = 0

    def __post_init__(self) -> None:
        if len(self.discs) != 19:
            raise ValueError(f"Nonaga must always contain exactly 19 discs, got {len(self.discs)}")
        if len(set(self.red)) != 3 or len(set(self.black)) != 3:
            raise ValueError("Each player must have three distinct token positions")
        occupied = set(self.red) | set(self.black)
        if len(occupied) != 6:
            raise ValueError("Token positions must be pairwise distinct")
        if not occupied.issubset(self.discs):
            raise ValueError("All tokens must stand on discs")
        if self.forbidden_disc is not None and self.forbidden_disc not in self.discs:
            raise ValueError("Forbidden disc must exist on the board")

    @property
    def occupied_tokens(self) -> set[Coord]:
        return set(self.red) | set(self.black)

    def pieces(self, color: Color) -> tuple[Coord, Coord, Coord]:
        return self.red if color == RED else self.black

    def normalized(self) -> GameState:
        anchor = min(self.discs)
        aq, ar = anchor

        def shift(c: Coord | None) -> Coord | None:
            if c is None:
                return None
            return (c[0] - aq, c[1] - ar)

        discs = frozenset((q - aq, r - ar) for q, r in self.discs)
        red = tuple(sorted((q - aq, r - ar) for q, r in self.red))  # type: ignore[assignment]
        black = tuple(sorted((q - aq, r - ar) for q, r in self.black))  # type: ignore[assignment]
        return GameState(discs, red, black, self.side_to_move, shift(self.forbidden_disc), self.ply)

    @property
    def canonical_key(self) -> tuple:
        """Translation- and symmetry-invariant key for transposition tables."""
        best: tuple | None = None
        for t in range(12):
            transformed_discs = [apply_transform(c, t) for c in self.discs]
            anchor = min(transformed_discs)
            aq, ar = anchor
            discs = normalize_translation(transformed_discs)
            red = normalize_translation(apply_transform(c, t) for c in self.red)
            black = normalize_translation(apply_transform(c, t) for c in self.black)
            forbidden = None
            if self.forbidden_disc is not None:
                transformed_forbidden = apply_transform(self.forbidden_disc, t)
                forbidden = (transformed_forbidden[0] - aq, transformed_forbidden[1] - ar)
            candidate = (self.side_to_move, discs, red, black, forbidden)
            if best is None or candidate < best:
                best = candidate
        assert best is not None
        return best

    @staticmethod
    def initial(side_to_move: Color = RED) -> GameState:
        discs = hex_radius_two()
        red = (CORNER_COORDS[0], CORNER_COORDS[2], CORNER_COORDS[4])
        black = (CORNER_COORDS[1], CORNER_COORDS[3], CORNER_COORDS[5])
        return GameState(discs, red, black, side_to_move, None, 0).normalized()
