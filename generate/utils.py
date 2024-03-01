from __future__ import annotations

import asyncio
from typing import AsyncIterator, Awaitable, Generator, Generic, Iterable, TypeVar

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


def sync_await(awaitable: Awaitable[T]) -> T:
    loop = asyncio.get_event_loop()
    try:
        return loop.run_until_complete(awaitable)
    except RuntimeError:
        import nest_asyncio

        nest_asyncio.apply()
        return loop.run_until_complete(awaitable)


def sync_aiter(aiterator: AsyncIterator[T]) -> Generator[T, None, None]:
    loop = asyncio.get_event_loop()
    try:
        while True:
            yield loop.run_until_complete(aiterator.__anext__())
    except StopAsyncIteration:
        return
    except RuntimeError:
        import nest_asyncio

        nest_asyncio.apply()
        yield from sync_aiter(aiterator)
