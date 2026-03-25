from __future__ import annotations

from dataclasses import dataclass

from .geometry import Coord


@dataclass(frozen=True, slots=True)
class PieceSlide:
    piece_index: int
    start: Coord
    end: Coord
    direction: int


@dataclass(frozen=True, slots=True)
class TileRelocation:
    remove_from: Coord
    place_to: Coord


@dataclass(frozen=True, slots=True)
class CompoundMove:
    slide: PieceSlide
    tile: TileRelocation

    def short(self) -> str:
        return f"p{self.slide.piece_index}:{self.slide.start}->{self.slide.end} | tile:{self.tile.remove_from}->{self.tile.place_to}"
