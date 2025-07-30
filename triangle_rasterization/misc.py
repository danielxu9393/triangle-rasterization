from typing import TypeVar

T = TypeVar("T")


def ceildiv(numerator: T, denominator: T) -> T:
    return (numerator + denominator - 1) // denominator


NEAR_PLANE = 0.2
TILE_KEYS = 1_000
