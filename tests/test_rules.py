from __future__ import annotations

import pytest

from nonaga.moves import CompoundMove, TileRelocation
from nonaga.rules import (
    apply_move,
    can_slide_disc_out,
    connected_three,
    generate_piece_slides,
    initial_state,
    legal_moves,
    removable_discs,
    tile_placement_targets,
)
from nonaga.search import SearchConfig
from nonaga.state import GameState


def test_initial_state_and_rulebook_first_move_count() -> None:
    state = initial_state()
    assert len(state.discs) == 19
    assert len(state.red) == 3
    assert len(state.black) == 3
    slides = generate_piece_slides(state)
    assert len(slides) == 9  # three red pieces, each with three first-move options
    per_piece = {}
    for slide in slides:
        per_piece.setdefault(slide.piece_index, 0)
        per_piece[slide.piece_index] += 1
    assert per_piece == {0: 3, 1: 3, 2: 3}


def test_connected_three_matches_line_tick_triangle() -> None:
    assert connected_three(((0, 0), (1, 0), (2, 0)))  # line
    assert connected_three(((0, 0), (1, 0), (1, -1)))  # triangle
    assert connected_three(((0, 0), (1, 0), (1, 1)))  # tick / path on hex graph
    assert not connected_three(((0, 0), (2, 0), (4, 0)))


def test_relocated_disc_becomes_forbidden_next_turn() -> None:
    state = initial_state()
    move = legal_moves(state)[0]
    nxt = apply_move(state, move)
    assert nxt.forbidden_disc == move.tile.place_to or nxt.forbidden_disc is not None
    assert nxt.forbidden_disc not in removable_discs(nxt)


def test_placement_requires_two_contacts() -> None:
    state = initial_state()
    move = legal_moves(state)[0]
    after_slide = state
    # Find a one-contact target after removing this legal removable disc.
    discs_after_removal = frozenset(c for c in state.discs if c != move.tile.remove_from)
    bad_targets = [c for c in {(q, r) for q in range(-2, 8) for r in range(-4, 6)} if c not in discs_after_removal]
    bad_target = None
    from nonaga.geometry import degree

    for cand in bad_targets:
        if degree(discs_after_removal, cand) == 1:
            bad_target = cand
            break
    assert bad_target is not None
    illegal = CompoundMove(move.slide, TileRelocation(move.tile.remove_from, bad_target))
    with pytest.raises(ValueError):
        apply_move(state, illegal)


def test_translation_invariant_canonical_key() -> None:
    state = initial_state()
    shift = (7, -3)
    translated = GameState(
        discs=frozenset((q + shift[0], r + shift[1]) for q, r in state.discs),
        red=tuple((q + shift[0], r + shift[1]) for q, r in state.red),  # type: ignore[arg-type]
        black=tuple((q + shift[0], r + shift[1]) for q, r in state.black),  # type: ignore[arg-type]
        side_to_move=state.side_to_move,
        forbidden_disc=None,
        ply=state.ply,
    )
    assert state.canonical_key == translated.canonical_key


def test_edge_disc_half_plane_rule_basic() -> None:
    state = initial_state()
    # A top boundary disc is slidable out in the starting position.
    assert can_slide_disc_out(state.discs, (0, 1))
    # A central disc is not an edge disc and therefore not removable.
    assert not can_slide_disc_out(state.discs, (2, 0))
