from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .geometry import degree, pairwise_path_sum
from .rules import generate_piece_slides, winner
from .state import Color, GameState, other_color


WIN_SCORE = 1_000_000.0
DIRS = {(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)}


@dataclass
class EvalWeights:
    edge_count: float = 180.0
    path_sum: float = -20.0
    piece_degree: float = 12.0
    slide_mobility: float = 3.0

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @staticmethod
    def from_json(path: str | Path) -> "EvalWeights":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return EvalWeights(**data)


def adjacency_edges(coords: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]) -> int:
    a, b, c = coords
    edges = 0
    for x, y in ((a, b), (a, c), (b, c)):
        if (y[0] - x[0], y[1] - x[1]) in DIRS:
            edges += 1
    return edges


def side_features(state: GameState, color: Color) -> dict[str, float]:
    pieces = state.pieces(color)
    proxy = GameState(state.discs, state.red, state.black, color, state.forbidden_disc, state.ply)
    return {
        "edge_count": float(adjacency_edges(pieces)),
        "path_sum": float(pairwise_path_sum(state.discs, pieces)),
        "piece_degree": float(sum(degree(state.discs, p) for p in pieces)),
        "slide_mobility": float(len(generate_piece_slides(proxy, color=color))),
    }


def evaluate(state: GameState, perspective: Color, weights: EvalWeights | None = None, previous_player: Color | None = None) -> float:
    weights = weights or EvalWeights()
    won = winner(state, previous_player=previous_player)
    if won == perspective:
        return WIN_SCORE - state.ply
    if won == other_color(perspective):
        return -WIN_SCORE + state.ply

    own = side_features(state, perspective)
    opp = side_features(state, other_color(perspective))
    score = 0.0
    score += weights.edge_count * (own["edge_count"] - opp["edge_count"])
    score += weights.path_sum * (own["path_sum"] - opp["path_sum"])
    score += weights.piece_degree * (own["piece_degree"] - opp["piece_degree"])
    score += weights.slide_mobility * (own["slide_mobility"] - opp["slide_mobility"])
    return score
