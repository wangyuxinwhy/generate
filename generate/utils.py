from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Generator, Generic, Iterable, TypeVar
from urllib.parse import urlparse

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


def unwrap_model(model: Any) -> Any:
    from generate.model import GenerateModel

    if hasattr(model, 'model'):
        if isinstance(model.model, GenerateModel):
            return unwrap_model(model.model)
        return model
    return model


def fetch_data(url_or_file: str) -> bytes:
    parsed_url = urlparse(url_or_file)
    if not parsed_url.scheme and Path(url_or_file).exists():
        return Path(url_or_file).read_bytes()

    if parsed_url.scheme == 'file':
        if Path(parsed_url.path).exists():
            return Path(parsed_url.path).read_bytes()
        raise FileNotFoundError(f'File {parsed_url.path} not found')

    if parsed_url.scheme in ('http', 'https'):
        import httpx

        response = httpx.get(url_or_file)
        response.raise_for_status()
        return response.content

    raise ValueError(f'Unsupported URL scheme {parsed_url.scheme}')
