from __future__ import annotations

import random
from dataclasses import dataclass, field

from .eval import EvalWeights, evaluate
from .moves import CompoundMove
from .rules import apply_move, legal_moves
from .search import SearchConfig, SearchResult, Searcher
from .state import GameState


class Agent:
    name: str = "agent"

    def choose_move(self, state: GameState) -> CompoundMove:
        raise NotImplementedError


@dataclass(slots=True)
class RandomAgent(Agent):
    seed: int | None = None
    name: str = "random"

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)

    def choose_move(self, state: GameState) -> CompoundMove:
        moves = legal_moves(state)
        if not moves:
            raise RuntimeError("No legal moves available")
        return self.rng.choice(moves)


@dataclass(slots=True)
class GreedyAgent(Agent):
    weights: EvalWeights = field(default_factory=EvalWeights)
    seed: int | None = None
    name: str = "greedy"

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)

    def choose_move(self, state: GameState) -> CompoundMove:
        moves = legal_moves(state)
        if not moves:
            raise RuntimeError("No legal moves available")
        scored: list[tuple[float, CompoundMove]] = []
        for mv in moves:
            nxt = apply_move(state, mv)
            scored.append((evaluate(nxt, state.side_to_move, self.weights, previous_player=state.side_to_move), mv))
        best_score = max(score for score, _ in scored)
        best = [mv for score, mv in scored if score == best_score]
        return self.rng.choice(best)


@dataclass(slots=True)
class SearchAgent(Agent):
    config: SearchConfig = field(default_factory=SearchConfig)
    weights: EvalWeights = field(default_factory=EvalWeights)
    name: str = "search"

    def __post_init__(self) -> None:
        self.searcher = Searcher(config=self.config, weights=self.weights)
        self.last_result: SearchResult | None = None

    def choose_move(self, state: GameState) -> CompoundMove:
        result = self.searcher.choose_move(state)
        self.last_result = result
        if result.move is None:
            raise RuntimeError("Search could not find a move")
        return result.move
