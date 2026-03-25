from __future__ import annotations

from nonaga.agents import RandomAgent, SearchAgent
from nonaga.rules import legal_moves
from nonaga.search import SearchConfig
from nonaga.selfplay import play_game
from nonaga.state import GameState


def test_search_agent_returns_legal_move() -> None:
    state = GameState.initial()
    agent = SearchAgent(config=SearchConfig(max_depth=1, time_limit=None, max_branching=20))
    move = agent.choose_move(state)
    assert move in legal_moves(state)
    assert agent.last_result is not None
    assert agent.last_result.depth_reached == 1


def test_random_vs_random_game_finishes_under_ply_cap() -> None:
    result = play_game(RandomAgent(seed=1), RandomAgent(seed=2), max_ply=20, repetition_draw=2)
    assert result.plies <= 20
    assert result.termination in {"win", "max_ply", "repetition x2", "no_legal_moves"}
