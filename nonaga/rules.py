from __future__ import annotations

from itertools import chain
from typing import Iterable

from .geometry import (
    Coord,
    add,
    all_rays,
    connected,
    degree,
    empty_cells_adjacent_to_board,
    neighbors,
    occupied_neighbor_dirs,
)
from .moves import CompoundMove, PieceSlide, TileRelocation
from .state import BLACK, RED, Color, GameState, other_color


def initial_state(side_to_move: Color = RED) -> GameState:
    return GameState.initial(side_to_move)


def are_adjacent(a: Coord, b: Coord) -> bool:
    return b in neighbors(a)


# The official rules show three winning patterns: line, tick, triangle.
# Those are exactly the three connected graphs on three tokens: either a path
# (2 adjacency edges) or a triangle (3 adjacency edges).
def connected_three(coords: tuple[Coord, Coord, Coord]) -> bool:
    a, b, c = coords
    edges = 0
    edges += int(are_adjacent(a, b))
    edges += int(are_adjacent(a, c))
    edges += int(are_adjacent(b, c))
    return edges >= 2


def winner(state: GameState, previous_player: Color | None = None) -> Color | None:
    red_win = connected_three(state.red)
    black_win = connected_three(state.black)
    if red_win and black_win:
        # Rules are silent on simultaneous wins. In practice, only the last mover
        # can create a new terminal position; favor that player deterministically.
        return previous_player
    if red_win:
        return RED
    if black_win:
        return BLACK
    return None


def is_terminal(state: GameState, previous_player: Color | None = None, max_ply: int | None = None) -> bool:
    if max_ply is not None and state.ply >= max_ply:
        return True
    return winner(state, previous_player=previous_player) is not None


def generate_piece_slides(state: GameState, color: Color | None = None) -> list[PieceSlide]:
    color = state.side_to_move if color is None else color
    own = state.pieces(color)
    occupied = state.occupied_tokens
    slides: list[PieceSlide] = []
    for idx, pos in enumerate(own):
        blocked = occupied - {pos}
        for dir_index, dst in all_rays(state.discs, blocked, pos):
            slides.append(PieceSlide(idx, pos, dst, dir_index))
    return slides


def is_edge_disc(discs: frozenset[Coord], coord: Coord) -> bool:
    return any(n not in discs for n in neighbors(coord))


def can_slide_disc_out(discs: frozenset[Coord], coord: Coord) -> bool:
    """
    Approximate the official 'slide out without moving another disc' rule by a
    half-plane test on touching neighbors.

    For equal circles on a hex packing, a disc can translate away if all discs
    touching it lie within some closed 180-degree half-plane. On the discrete
    axial grid this is equivalent to the occupied neighbor directions fitting
    inside four consecutive of the six hex directions.
    """
    if coord not in discs or not is_edge_disc(discs, coord):
        return False
    occ = occupied_neighbor_dirs(discs, coord)
    if not occ:
        return True
    for start in range(6):
        allowed = {(start + k) % 6 for k in range(4)}
        if all(i in allowed for i in occ):
            return True
    return False


def removable_discs(state: GameState) -> list[Coord]:
    occupied = state.occupied_tokens
    out: list[Coord] = []
    for coord in state.discs:
        if coord in occupied:
            continue
        if state.forbidden_disc is not None and coord == state.forbidden_disc:
            continue
        if can_slide_disc_out(state.discs, coord):
            out.append(coord)
    return sorted(out)


def tile_placement_targets(discs_after_removal: frozenset[Coord], removed_coord: Coord) -> list[Coord]:
    out: list[Coord] = []
    for cell in empty_cells_adjacent_to_board(discs_after_removal):
        if cell == removed_coord:
            continue
        if degree(discs_after_removal, cell) >= 2:
            out.append(cell)
    return sorted(out)


def apply_piece_slide(state: GameState, slide: PieceSlide) -> GameState:
    own = list(state.pieces(state.side_to_move))
    if own[slide.piece_index] != slide.start:
        raise ValueError("PieceSlide start/index mismatch")
    own[slide.piece_index] = slide.end
    new_own = tuple(sorted(own))  # type: ignore[assignment]
    if state.side_to_move == RED:
        return GameState(state.discs, new_own, state.black, state.side_to_move, state.forbidden_disc, state.ply)
    return GameState(state.discs, state.red, new_own, state.side_to_move, state.forbidden_disc, state.ply)



def apply_move(state: GameState, move: CompoundMove) -> GameState:
    after_slide = apply_piece_slide(state, move.slide)
    if move.tile.remove_from in after_slide.occupied_tokens:
        raise ValueError("Cannot remove a disc under a token")
    if after_slide.forbidden_disc is not None and move.tile.remove_from == after_slide.forbidden_disc:
        raise ValueError("Cannot remove the disc relocated on the previous turn")
    if not can_slide_disc_out(after_slide.discs, move.tile.remove_from):
        raise ValueError("Removed disc is not legally slidable out")

    discs_after_removal = frozenset(c for c in after_slide.discs if c != move.tile.remove_from)
    if move.tile.place_to in discs_after_removal:
        raise ValueError("Placement target already occupied by another disc")
    if move.tile.place_to == move.tile.remove_from:
        raise ValueError("The relocated disc must move to another position")
    if degree(discs_after_removal, move.tile.place_to) < 2:
        raise ValueError("Relocated disc must contact at least two other discs")

    new_discs = frozenset(chain(discs_after_removal, [move.tile.place_to]))
    if len(new_discs) != 19:
        raise ValueError("A move must preserve the total number of discs")

    if state.side_to_move == RED:
        new_state = GameState(new_discs, after_slide.red, state.black, other_color(state.side_to_move), move.tile.place_to, state.ply + 1)
    else:
        new_state = GameState(new_discs, state.red, after_slide.black, other_color(state.side_to_move), move.tile.place_to, state.ply + 1)
    return new_state.normalized()


def legal_moves_from_slide(state_after_slide: GameState, slide: PieceSlide) -> list[CompoundMove]:
    moves: list[CompoundMove] = []
    for remove_from in removable_discs(state_after_slide):
        discs_after_removal = frozenset(c for c in state_after_slide.discs if c != remove_from)
        for place_to in tile_placement_targets(discs_after_removal, remove_from):
            moves.append(CompoundMove(slide, TileRelocation(remove_from, place_to)))
    return moves


def legal_moves(state: GameState) -> list[CompoundMove]:
    moves: list[CompoundMove] = []
    for slide in generate_piece_slides(state):
        after_slide = apply_piece_slide(state, slide)
        moves.extend(legal_moves_from_slide(after_slide, slide))
    return moves


def immediate_winning_moves(state: GameState, color: Color | None = None) -> list[CompoundMove]:
    color = state.side_to_move if color is None else color
    if state.side_to_move != color:
        # Make a shallow proxy state with the desired side to move.
        state = GameState(state.discs, state.red, state.black, color, state.forbidden_disc, state.ply)
    out: list[CompoundMove] = []
    for move in legal_moves(state):
        nxt = apply_move(state, move)
        if winner(nxt, previous_player=color) == color:
            out.append(move)
    return out


def repetition_key(state: GameState) -> tuple:
    return state.canonical_key


def assert_legal_state(state: GameState) -> None:
    assert len(state.discs) == 19
    assert len(state.occupied_tokens) == 6
    assert state.red and state.black
