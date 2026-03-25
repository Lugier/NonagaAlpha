from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import NamedTuple

from .eval import EvalWeights, evaluate
from .moves import CompoundMove
from .rules import apply_move, is_terminal, legal_moves, repetition_key, winner
from .state import Color, GameState, other_color


INF = 10_000_000.0


class SearchResult(NamedTuple):
    move: CompoundMove | None
    score: float
    depth_reached: int
    nodes: int
    elapsed: float


@dataclass
class SearchConfig:
    max_depth: int = 3
    time_limit: float | None = 1.0
    max_branching: int = 80
    max_ply: int = 160
    repetition_draw: int = 3
    use_symmetry_tt: bool = True


@dataclass
class TTEntry:
    depth: int
    score: float
    flag: str
    best_move: CompoundMove | None


@dataclass
class Searcher:
    config: SearchConfig = field(default_factory=SearchConfig)
    weights: EvalWeights = field(default_factory=EvalWeights)
    tt: dict[tuple, TTEntry] = field(default_factory=dict, init=False)
    history: dict[CompoundMove, float] = field(default_factory=dict, init=False)
    killers: dict[int, list[CompoundMove]] = field(default_factory=dict, init=False)
    nodes: int = field(default=0, init=False)
    deadline: float | None = field(default=None, init=False)
    root_color: Color | None = field(default=None, init=False)

    def choose_move(self, state: GameState) -> SearchResult:
        self.nodes = 0
        self.root_color = state.side_to_move
        start = time.perf_counter()
        self.deadline = None if self.config.time_limit is None else start + self.config.time_limit
        legal = legal_moves(state)
        if not legal:
            elapsed = time.perf_counter() - start
            return SearchResult(None, -INF, 0, self.nodes, elapsed)
        best_move: CompoundMove | None = legal[0]
        best_score = evaluate(apply_move(state, best_move), state.side_to_move, self.weights, previous_player=state.side_to_move)
        history_counts = {repetition_key(state): 1}
        depth_reached = 0

        for depth in range(1, self.config.max_depth + 1):
            if self._timed_out():
                break
            score, move = self._negamax_root(state, depth, history_counts)
            if move is not None:
                best_move = move
                best_score = score
                depth_reached = depth
            if self._timed_out():
                break
        elapsed = time.perf_counter() - start
        return SearchResult(best_move, best_score, depth_reached, self.nodes, elapsed)

    def _timed_out(self) -> bool:
        return self.deadline is not None and time.perf_counter() >= self.deadline

    def _key(self, state: GameState) -> tuple:
        return state.canonical_key if self.config.use_symmetry_tt else (state.side_to_move, state.discs, state.red, state.black, state.forbidden_disc)

    def _child_order(self, state: GameState, depth: int) -> list[tuple[float, CompoundMove, GameState]]:
        perspective = self.root_color or state.side_to_move
        children: list[tuple[float, CompoundMove, GameState]] = []
        moves = legal_moves(state)
        for move in moves:
            nxt = apply_move(state, move)
            score = evaluate(nxt, perspective, self.weights, previous_player=state.side_to_move)
            score += self.history.get(move, 0.0)
            killers = self.killers.get(depth, [])
            if move in killers:
                score += 5000.0
            if winner(nxt, previous_player=state.side_to_move) == state.side_to_move:
                score += 100_000.0
            children.append((score, move, nxt))
        reverse = state.side_to_move == perspective
        children.sort(key=lambda item: item[0], reverse=reverse)
        if self.config.max_branching and len(children) > self.config.max_branching:
            children = children[: self.config.max_branching]
        return children

    def _negamax_root(self, state: GameState, depth: int, history_counts: dict[tuple, int]) -> tuple[float, CompoundMove | None]:
        alpha = -INF
        beta = INF
        best_move: CompoundMove | None = None
        perspective = self.root_color or state.side_to_move
        ordered = self._child_order(state, depth)
        if not ordered:
            return evaluate(state, perspective, self.weights), None
        for _, move, nxt in ordered:
            if self._timed_out():
                break
            key = repetition_key(nxt)
            history_counts[key] = history_counts.get(key, 0) + 1
            score = -self._negamax(
                nxt,
                depth - 1,
                -beta,
                -alpha,
                history_counts,
                previous_player=state.side_to_move,
            )
            history_counts[key] -= 1
            if history_counts[key] == 0:
                del history_counts[key]
            if score > alpha:
                alpha = score
                best_move = move
        return alpha, best_move

    def _negamax(
        self,
        state: GameState,
        depth: int,
        alpha: float,
        beta: float,
        history_counts: dict[tuple, int],
        previous_player: Color | None,
    ) -> float:
        self.nodes += 1
        perspective = self.root_color or state.side_to_move
        if self._timed_out():
            return evaluate(state, perspective, self.weights, previous_player=previous_player)

        if history_counts.get(repetition_key(state), 0) >= self.config.repetition_draw:
            return 0.0
        if is_terminal(state, previous_player=previous_player, max_ply=self.config.max_ply):
            won = winner(state, previous_player=previous_player)
            return 0.0 if won is None else evaluate(state, perspective, self.weights, previous_player=previous_player)
        if depth <= 0:
            return evaluate(state, perspective, self.weights, previous_player=previous_player)

        key = self._key(state)
        entry = self.tt.get(key)
        if entry and entry.depth >= depth:
            if entry.flag == "EXACT":
                return entry.score
            if entry.flag == "LOWER":
                alpha = max(alpha, entry.score)
            elif entry.flag == "UPPER":
                beta = min(beta, entry.score)
            if alpha >= beta:
                return entry.score

        alpha_orig = alpha
        best_score = -INF
        best_move: CompoundMove | None = entry.best_move if entry else None

        children = self._child_order(state, depth)
        if entry and entry.best_move is not None:
            for i, child in enumerate(children):
                if child[1] == entry.best_move:
                    children.insert(0, children.pop(i))
                    break

        if not children:
            return evaluate(state, perspective, self.weights, previous_player=previous_player)

        for _, move, nxt in children:
            child_key = repetition_key(nxt)
            history_counts[child_key] = history_counts.get(child_key, 0) + 1
            score = -self._negamax(
                nxt,
                depth - 1,
                -beta,
                -alpha,
                history_counts,
                previous_player=state.side_to_move,
            )
            history_counts[child_key] -= 1
            if history_counts[child_key] == 0:
                del history_counts[child_key]

            if score > best_score:
                best_score = score
                best_move = move
            alpha = max(alpha, score)
            if alpha >= beta:
                self.history[move] = self.history.get(move, 0.0) + depth * depth
                killers = self.killers.setdefault(depth, [])
                if move not in killers:
                    killers.insert(0, move)
                    del killers[2:]
                break

        flag = "EXACT"
        if best_score <= alpha_orig:
            flag = "UPPER"
        elif best_score >= beta:
            flag = "LOWER"
        self.tt[key] = TTEntry(depth=depth, score=best_score, flag=flag, best_move=best_move)
        return best_score
