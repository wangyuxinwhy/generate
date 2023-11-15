import logging
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Generic, TypeVar

from typing_extensions import Self

from generate.image_generation.model_output import ImageGenerationOutput
from generate.model import ModelInfo
from generate.parameters import ModelParameters

P = TypeVar('P', bound=ModelParameters)
logger = logging.getLogger(__name__)


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
    def _image_generation(self, prompt: str, parameters: P) -> ImageGenerationOutput:
        ...

    @abstractmethod
    async def _async_image_generation(self, prompt: str, parameters: P) -> ImageGenerationOutput:
        ...

    @property
    def model_info(self) -> ModelInfo:
        return ModelInfo(
            task='image_generation',
            type=self.model_type,
            name=self.name,
        )

    def generate(self, prompt: str, **override_parameters: Any) -> ImageGenerationOutput:
        parameters = self._merge_parameters(**override_parameters)
        logger.debug(f'{prompt=}, {parameters=}')
        return self._image_generation(prompt, parameters)

    async def async_generate(self, prompt: str, **override_parameters: Any) -> ImageGenerationOutput:
        parameters = self._merge_parameters(**override_parameters)
        logger.debug(f'{prompt=}, {parameters=}')
        return await self._async_image_generation(prompt, parameters)

    def _merge_parameters(self, **override_parameters: Any) -> P:
        return self.parameters.__class__.model_validate(
            {**self.parameters.model_dump(exclude_unset=True), **override_parameters}
        )
