from __future__ import annotations

from typing import Iterable

from .geometry import Coord


Cube = tuple[int, int, int]


def axial_to_cube(c: Coord) -> Cube:
    q, r = c
    return (q, -q - r, r)


def cube_to_axial(c: Cube) -> Coord:
    x, _, z = c
    return (x, z)


def rotate_left(c: Coord) -> Coord:
    x, y, z = axial_to_cube(c)
    return cube_to_axial((-z, -x, -y))


def reflect_q(c: Coord) -> Coord:
    q, r = c
    return (q, -q - r)


def apply_transform(c: Coord, idx: int) -> Coord:
    assert 0 <= idx < 12
    out = c
    if idx >= 6:
        out = reflect_q(out)
    for _ in range(idx % 6):
        out = rotate_left(out)
    return out


def normalize_translation(coords: Iterable[Coord]) -> tuple[Coord, ...]:
    pts = list(coords)
    if not pts:
        return tuple()
    anchor = min(pts)
    aq, ar = anchor
    shifted = sorted((q - aq, r - ar) for q, r in pts)
    return tuple(shifted)
