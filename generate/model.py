from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Dict, Generic, Optional, TypeVar

from pydantic import BaseModel, ConfigDict
from typing_extensions import Self, TypedDict, Unpack


class ModelParameters(BaseModel):
    def custom_model_dump(self) -> dict[str, Any]:
        return {**self.model_dump(exclude_none=True, by_alias=True), **self.model_dump(exclude_unset=True, by_alias=True)}

    def update_with_validate(self, **kwargs: Any) -> Self:
        return self.__class__.model_validate({**self.model_dump(exclude_unset=True), **kwargs})  # type: ignore


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


I = TypeVar('I', bound=Any)  # noqa: E741
O = TypeVar('O', bound=ModelOutput)  # noqa: E741


class GenerateModel(Generic[I, O], ABC):
    model_task: ClassVar[str]
    model_type: ClassVar[str]

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @classmethod
    @abstractmethod
    def from_name(cls, name: str) -> Self:
        ...

    @abstractmethod
    def generate(self, prompt: I, **kwargs: Unpack[ModelParametersDict]) -> O:
        ...

    @abstractmethod
    async def async_generate(self, prompt: I, **kwargs: Unpack[ModelParametersDict]) -> O:
        ...

    @property
    def model_info(self) -> ModelInfo:
        return ModelInfo(task=self.model_task, type=self.model_type, name=self.name)

    @property
    def model_id(self) -> str:
        return self.model_info.model_id
