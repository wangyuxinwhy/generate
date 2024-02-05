from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, AsyncGenerator, Dict, Generator, Generic, Iterable, Optional, TypeVar

import anyio
import asyncer
from pydantic import BaseModel, ConfigDict
from typing_extensions import Self, TypedDict

if TYPE_CHECKING:
    from generate.modifiers.limit import Limit


class ModelParameters(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    def custom_model_dump(self) -> dict[str, Any]:
        return {**self.model_dump(exclude_none=True, by_alias=True), **self.model_dump(exclude_unset=True, by_alias=True)}

    def clone_with_changes(self, **changes: Any) -> Self:
        return self.__class__.model_validate({**self.model_dump(exclude_unset=True), **changes})  # type: ignore

    def model_update(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)


class ModelParametersDict(TypedDict, total=False):
    ...


class ModelInfo(BaseModel):
    task: str
    type: str
    name: str

    @property
    def model_id(self) -> str:
        return f'{self.type}/{self.name}'


class ModelOutput(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_info: ModelInfo
    cost: Optional[float] = None
    extra: Dict[str, Any] = {}


I = TypeVar('I')  # noqa: E741
O = TypeVar('O', bound=ModelOutput)  # noqa: E741


class GenerateModel(Generic[I, O], ABC):
    model_task: str
    model_type: str

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @classmethod
    @abstractmethod
    def from_name(cls, name: str) -> Self:
        ...

    @abstractmethod
    def generate(self, prompt: I, **kwargs: Any) -> O:
        ...

    @abstractmethod
    async def async_generate(self, prompt: I, **kwargs: Any) -> O:
        ...

    def batch_generate(self, prompts: Iterable[I], **kwargs: Any) -> Generator[O, None, None]:
        for prompt in prompts:
            yield self.generate(prompt, **kwargs)

    async def async_batch_generate(self, prompts: Iterable[I], **kwargs: Any) -> AsyncGenerator[O, None]:
        async with asyncer.create_task_group() as task_group:
            soon_values: list[asyncer.SoonValue[O]] = []
            for prompt in prompts:
                soon_value = task_group.soonify(self.async_generate)(prompt, **kwargs)
                soon_values.append(soon_value)
            for soon_value in soon_values:
                while not soon_value.ready:
                    await anyio.sleep(0.01)
                yield soon_value.value

    @property
    def model_info(self) -> ModelInfo:
        return ModelInfo(task=self.model_task, type=self.model_type, name=self.name)

    @property
    def model_id(self) -> str:
        return self.model_info.model_id

    def limit(
        self, async_capacity: int = 3, max_generates_per_time_window: int = 20, num_seconds_in_time_window: int = 60
    ) -> Limit[I, O]:
        from generate.modifiers.limit import Limit

        return Limit(
            self,
            async_capacity=async_capacity,
            max_generates_per_time_window=max_generates_per_time_window,
            num_seconds_in_time_window=num_seconds_in_time_window,
        )
