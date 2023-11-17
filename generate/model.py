from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Dict, Generic, Optional, TypeVar

from pydantic import BaseModel, ConfigDict, model_serializer
from typing_extensions import Self


class ModelParameters(BaseModel):
    def custom_model_dump(self) -> dict[str, Any]:
        return {**self.model_dump(exclude_none=True, by_alias=True), **self.model_dump(exclude_unset=True, by_alias=True)}


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

    @model_serializer(mode='wrap')
    def ser_model(self, handler: Any) -> Any:  # noqa: ANN001
        output = handler(self)
        output.pop('debug')
        return output


P = TypeVar('P', bound=ModelParameters)
I = TypeVar('I', bound=Any)  # noqa: E741
O = TypeVar('O', bound=ModelOutput)  # noqa: E741


class GenerateModel(Generic[P, I, O], ABC):
    model_task: ClassVar[str]
    model_type: ClassVar[str]

    def __init__(self, parameters: P) -> None:
        self.parameters = parameters

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @classmethod
    @abstractmethod
    def from_name(cls, name: str, **kwargs: Any) -> Self:
        ...

    @abstractmethod
    def generate(self, prompt: I, **override_parameters: Any) -> O:
        ...

    @abstractmethod
    async def async_generate(self, prompt: I, **override_parameters: Any) -> O:
        ...

    @property
    def model_info(self) -> ModelInfo:
        return ModelInfo(task=self.model_task, type=self.model_type, name=self.name)

    @property
    def model_id(self) -> str:
        return self.model_info.model_id

    def _merge_parameters(self, **override_parameters: Any) -> P:
        return self.parameters.__class__.model_validate(
            {**self.parameters.model_dump(exclude_unset=True), **override_parameters}
        )
