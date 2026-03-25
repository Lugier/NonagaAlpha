from __future__ import annotations

from collections import deque
from itertools import combinations
from typing import Iterable, Iterator

Coord = tuple[int, int]

# Axial directions for pointy-top hex coordinates.
DIRS: tuple[Coord, ...] = (
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, 0),
    (-1, 1),
    (0, 1),
)

CORNER_COORDS: tuple[Coord, ...] = (
    (2, 0),
    (2, -2),
    (0, -2),
    (-2, 0),
    (-2, 2),
    (0, 2),
)


def add(a: Coord, b: Coord) -> Coord:
    return (a[0] + b[0], a[1] + b[1])


def sub(a: Coord, b: Coord) -> Coord:
    return (a[0] - b[0], a[1] - b[1])


def neighbors(c: Coord) -> tuple[Coord, ...]:
    return tuple(add(c, d) for d in DIRS)


def hex_radius_two() -> frozenset[Coord]:
    """19-cell starting hexagon."""
    cells: set[Coord] = set()
    radius = 2
    for q in range(-radius, radius + 1):
        for r in range(-radius, radius + 1):
            s = -q - r
            if max(abs(q), abs(r), abs(s)) <= radius:
                cells.add((q, r))
    return frozenset(cells)


def occupied_neighbor_dirs(discs: set[Coord] | frozenset[Coord], c: Coord) -> list[int]:
    result: list[int] = []
    for i, d in enumerate(DIRS):
        if add(c, d) in discs:
            result.append(i)
    return result


def degree(discs: set[Coord] | frozenset[Coord], c: Coord) -> int:
    return sum(1 for n in neighbors(c) if n in discs)


def connected(discs: set[Coord] | frozenset[Coord]) -> bool:
    if not discs:
        return True
    start = next(iter(discs))
    seen = {start}
    dq = deque([start])
    while dq:
        cur = dq.popleft()
        for n in neighbors(cur):
            if n in discs and n not in seen:
                seen.add(n)
                dq.append(n)
    return len(seen) == len(discs)


def shortest_path_length(discs: frozenset[Coord], start: Coord, goal: Coord) -> int:
    if start == goal:
        return 0
    dq = deque([(start, 0)])
    seen = {start}
    while dq:
        cur, dist = dq.popleft()
        for n in neighbors(cur):
            if n not in discs or n in seen:
                continue
            if n == goal:
                return dist + 1
            seen.add(n)
            dq.append((n, dist + 1))
    return 99


def pairwise_path_sum(discs: frozenset[Coord], coords: Iterable[Coord]) -> int:
    pts = list(coords)
    total = 0
    for a, b in combinations(pts, 2):
        total += shortest_path_length(discs, a, b)
    return total


def empty_cells_adjacent_to_board(discs: frozenset[Coord]) -> set[Coord]:
    result: set[Coord] = set()
    for c in discs:
        for n in neighbors(c):
            if n not in discs:
                result.add(n)
    return result


def ray_until_stop(discs: frozenset[Coord], occupied_tokens: set[Coord], start: Coord, dir_index: int) -> Coord | None:
    """
    Slide across existing discs along one axial direction until the next step would
    leave the current field or would enter another token. The destination is the
    last reachable disc before that obstacle.
    """
    direction = DIRS[dir_index]
    cur = start
    nxt = add(cur, direction)
    if nxt not in discs or nxt in occupied_tokens:
        return None
    while nxt in discs and nxt not in occupied_tokens:
        cur = nxt
        nxt = add(cur, direction)
    return cur


def all_rays(discs: frozenset[Coord], occupied_tokens: set[Coord], start: Coord) -> Iterator[tuple[int, Coord]]:
    for i in range(6):
        dst = ray_until_stop(discs, occupied_tokens, start, i)
        if dst is not None and dst != start:
            yield i, dst
