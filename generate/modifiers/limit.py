import time
from itertools import islice
from typing import Any, AsyncGenerator, Generator, Iterable, List, TypeVar

import anyio
import asyncer
from typing_extensions import Self

from generate.model import GenerateModel, ModelOutput

I = TypeVar('I')  # noqa: E741
O = TypeVar('O', bound=ModelOutput)  # noqa: E741
T = TypeVar('T')


def _batch(items: Iterable[T], batch_size: int) -> Generator[List[T], None, None]:
    iterator = iter(items)
    while True:
        batch = list(islice(iterator, batch_size))
        if not batch:
            break
        yield batch


class Limit(GenerateModel[I, O]):
    def __init__(
        self,
        model: GenerateModel[I, O],
        async_capacity: int = 3,
        max_generates_per_time_window: int = 20,
        num_seconds_in_time_window: int = 60,
    ) -> None:
        self.model: GenerateModel[I, O] = model
        self.async_capacity = async_capacity
        self.max_generates_per_time_window = max_generates_per_time_window
        self.num_seconds_in_time_window = num_seconds_in_time_window
        assert self.num_seconds_in_time_window >= 1

        self.model_task = self.model.model_task
        self.model_type = self.model.model_type

        self._task_created_time_list: list[int] = []

    @property
    def name(self) -> str:
        return self.model.name

    @classmethod
    def from_name(cls, name: str) -> Self:
        raise ValueError('Limit model cannot be created from name')

    @property
    def limiter(self) -> anyio.CapacityLimiter:
        if hasattr(self, '_limiter'):
            return self._limiter
        self._limiter = anyio.CapacityLimiter(self.async_capacity)
        return self._limiter

    @property
    def task_created_lock(self) -> anyio.Lock:
        if hasattr(self, '_task_created_lock'):
            return self._task_created_lock
        self._task_created_lock = anyio.Lock()
        return self._task_created_lock

    def generate(self, prompt: I, **kwargs: Any) -> O:
        sleep_time = self._calculate_sleep_time()
        if sleep_time > 0:
            time.sleep(sleep_time)
        self._task_created_time_list.append(int(time.time()))
        return self.model.generate(prompt, **kwargs)

    async def async_generate(self, prompt: I, **kwargs: Any) -> O:
        async with self.limiter:
            async with self.task_created_lock:
                sleep_time = self._calculate_sleep_time()
                if sleep_time > 0:
                    await anyio.sleep(sleep_time)
                self._task_created_time_list.append(int(time.time()))
            return await self.model.async_generate(prompt, **kwargs)

    async def async_batch_generate(self, prompts: Iterable[I], **kwargs: Any) -> AsyncGenerator[O, None]:
        async with asyncer.create_task_group() as task_group:
            for batch_prompt in _batch(prompts, batch_size=self.async_capacity):
                soon_values: list[asyncer.SoonValue[O]] = []
                for prompt in batch_prompt:
                    soon_value = task_group.soonify(self.async_generate)(prompt, **kwargs)
                    soon_values.append(soon_value)
                for soon_value in soon_values:
                    while not soon_value.ready:
                        await anyio.sleep(0.01)
                    yield soon_value.value

    def _calculate_sleep_time(self) -> int:
        idx = 0
        current_time = time.time()
        for i, task_created_time in enumerate(self._task_created_time_list):
            if current_time - task_created_time < self.num_seconds_in_time_window:
                idx = i
                break
        self._task_created_time_list = self._task_created_time_list[idx:]

        if len(self._task_created_time_list) < self.max_generates_per_time_window:
            return 0

        return max(self.num_seconds_in_time_window - int(current_time - self._task_created_time_list[0]) + 1, 0)
