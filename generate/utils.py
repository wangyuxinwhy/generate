from __future__ import annotations

from typing import Generator, Generic, Iterable, TypeVar

from generate.types import OrIterable

T = TypeVar('T')
P = TypeVar('P')
R = TypeVar('R')


def ensure_iterable(obj: OrIterable[T]) -> Iterable[T]:
    if isinstance(obj, Iterable):
        return obj
    return [obj]


class CatchReturnGenerator(Generic[T, P, R]):
    value: R

    def __init__(self, generator: Generator[T, P, R]) -> None:
        self.generator = generator

    def __iter__(self) -> Generator[T, P, R]:
        self.value = yield from self.generator
        return self.value
