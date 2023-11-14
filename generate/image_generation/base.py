from abc import ABC, abstractmethod
from typing import Any, ClassVar, Generic, TypeVar

from typing_extensions import Self

from generate.image_generation.model_output import ImageGenerationModelOutput
from generate.parameters import ModelParameters

P = TypeVar('P', bound=ModelParameters)


class ImageGenerationModel(Generic[P], ABC):
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
    def _image_generation(self, prompt: str, parameters: P) -> ImageGenerationModelOutput:
        ...

    @abstractmethod
    async def _async_image_generation(self, prompt: str, parameters: P) -> ImageGenerationModelOutput:
        ...

    @property
    def model_id(self) -> str:
        return f'{self.model_type}/{self.name}'

    def generate(self, prompt: str, **override_parameters: Any) -> ImageGenerationModelOutput:
        parameters = self._merge_parameters(**override_parameters)
        model_output = self._image_generation(prompt, parameters)
        model_output.debug['prompt'] = prompt
        model_output.debug['parameters'] = parameters
        return model_output

    async def async_generate(self, prompt: str, **override_parameters: Any) -> ImageGenerationModelOutput:
        parameters = self._merge_parameters(**override_parameters)
        model_output = await self._async_image_generation(prompt, parameters)
        model_output.debug['prompt'] = prompt
        model_output.debug['parameters'] = parameters
        return model_output

    def _merge_parameters(self, **override_parameters: Any) -> P:
        return self.parameters.__class__.model_validate(
            {**self.parameters.model_dump(exclude_unset=True), **override_parameters}
        )
