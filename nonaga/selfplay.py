from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .agents import Agent
from .moves import CompoundMove
from .rules import apply_move, legal_moves, repetition_key, winner
from .state import BLACK, RED, Color, GameState, other_color


@dataclass(slots=True)
class GameResult:
    winner: Color | None
    termination: str
    plies: int
    moves: list[CompoundMove]
    final_state: GameState


@dataclass(slots=True)
class ArenaSummary:
    a_wins: int
    b_wins: int
    draws: int
    games: int

    @property
    def score_a(self) -> float:
        return self.a_wins + 0.5 * self.draws

    @property
    def win_rate_a(self) -> float:
        return self.score_a / self.games if self.games else 0.0


def play_game(
    red_agent: Agent,
    black_agent: Agent,
    *,
    start_state: GameState | None = None,
    max_ply: int = 160,
    repetition_draw: int = 3,
) -> GameResult:
    state = start_state or GameState.initial(RED)
    history_counts = {repetition_key(state): 1}
    moves: list[CompoundMove] = []
    previous_player: Color | None = None

    while True:
        if history_counts.get(repetition_key(state), 0) >= repetition_draw:
            return GameResult(None, f"repetition x{repetition_draw}", len(moves), moves, state)
        if state.ply >= max_ply:
            return GameResult(None, "max_ply", len(moves), moves, state)
        won = winner(state, previous_player=previous_player)
        if won is not None:
            return GameResult(won, "win", len(moves), moves, state)

        moves_now = legal_moves(state)
        if not moves_now:
            return GameResult(other_color(state.side_to_move), "no_legal_moves", len(moves), moves, state)

        agent = red_agent if state.side_to_move == RED else black_agent
        move = agent.choose_move(state)
        if move not in moves_now:
            raise ValueError(f"Agent {agent.name} produced an illegal move: {move}")
        state = apply_move(state, move)
        moves.append(move)
        previous_player = other_color(state.side_to_move)
        key = repetition_key(state)
        history_counts[key] = history_counts.get(key, 0) + 1


def arena(
    agent_a_factory: Callable[[], Agent],
    agent_b_factory: Callable[[], Agent],
    *,
    games: int = 20,
    max_ply: int = 160,
    repetition_draw: int = 3,
) -> ArenaSummary:
    a_wins = 0
    b_wins = 0
    draws = 0
    for i in range(games):
        if i % 2 == 0:
            result = play_game(agent_a_factory(), agent_b_factory(), max_ply=max_ply, repetition_draw=repetition_draw)
            if result.winner == RED:
                a_wins += 1
            elif result.winner == BLACK:
                b_wins += 1
            else:
                draws += 1
        else:
            result = play_game(agent_b_factory(), agent_a_factory(), max_ply=max_ply, repetition_draw=repetition_draw)
            if result.winner == RED:
                b_wins += 1
            elif result.winner == BLACK:
                a_wins += 1
            else:
                draws += 1
    return ArenaSummary(a_wins=a_wins, b_wins=b_wins, draws=draws, games=games)
