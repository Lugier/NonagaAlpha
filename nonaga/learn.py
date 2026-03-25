from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from .agents import SearchAgent
from .eval import EvalWeights
from .search import SearchConfig
from .selfplay import ArenaSummary, arena


@dataclass(slots=True)
class LearnConfig:
    generations: int = 20
    candidates_per_generation: int = 6
    arena_games: int = 12
    mutation_scale: float = 0.20
    seed: int = 0
    output_path: str = "best_weights.json"
    max_depth: int = 2
    time_limit: float | None = 0.25
    max_branching: int = 60


@dataclass(slots=True)
class LearnResult:
    weights: EvalWeights
    history: list[dict]


def _mutate(base: EvalWeights, rng: random.Random, scale: float) -> EvalWeights:
    data = asdict(base)
    out: dict[str, float] = {}
    for key, value in data.items():
        sigma = max(abs(value) * scale, scale)
        out[key] = float(value + rng.gauss(0.0, sigma))
    return EvalWeights(**out)


def _make_agent(weights: EvalWeights, cfg: LearnConfig) -> SearchAgent:
    search_cfg = SearchConfig(max_depth=cfg.max_depth, time_limit=cfg.time_limit, max_branching=cfg.max_branching)
    return SearchAgent(config=search_cfg, weights=weights)


def tune(initial: EvalWeights | None = None, config: LearnConfig | None = None) -> LearnResult:
    cfg = config or LearnConfig()
    rng = random.Random(cfg.seed)
    champion = initial or EvalWeights()
    history: list[dict] = []

    output_path = Path(cfg.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for gen in range(cfg.generations):
        best_candidate = champion
        best_summary: ArenaSummary | None = None
        best_score = -math.inf
        for _ in range(cfg.candidates_per_generation):
            candidate = _mutate(champion, rng, cfg.mutation_scale)
            summary = arena(
                agent_a_factory=lambda c=candidate: _make_agent(c, cfg),
                agent_b_factory=lambda c=champion: _make_agent(c, cfg),
                games=cfg.arena_games,
            )
            score = summary.score_a - (cfg.arena_games / 2)
            if score > best_score:
                best_candidate = candidate
                best_summary = summary
                best_score = score

        if best_summary is not None and best_summary.win_rate_a > 0.5:
            champion = best_candidate

        record = {
            "generation": gen,
            "weights": asdict(champion),
            "candidate_score": best_score,
            "summary": None if best_summary is None else asdict(best_summary),
        }
        history.append(record)
        output_path.write_text(json.dumps(asdict(champion), indent=2), encoding="utf-8")

    return LearnResult(weights=champion, history=history)
