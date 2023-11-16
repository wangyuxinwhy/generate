import logging
from abc import ABC, abstractmethod
from typing import Any, ClassVar, List, Optional, TypeVar

from pydantic import BaseModel

from generate.model import GenerateModel, ModelOutput, ModelParameters

P = TypeVar('P', bound=ModelParameters)
logger = logging.getLogger(__name__)


class GeneratedImage(BaseModel):
    url: Optional[str] = None
    prompt: str
    image_format: str
    content: bytes


class ImageGenerationOutput(ModelOutput):
    images: List[GeneratedImage] = []


class ImageGenerationModel(GenerateModel[P, str, ImageGenerationOutput], ABC):
    model_task: ClassVar[str] = 'image_generation'
    model_type: ClassVar[str]

    def __init__(self, parameters: P) -> None:
        self.parameters = parameters

    @abstractmethod
    def _image_generation(self, prompt: str, parameters: P) -> ImageGenerationOutput:
        ...

    @abstractmethod
    async def _async_image_generation(self, prompt: str, parameters: P) -> ImageGenerationOutput:
        ...

    def generate(self, prompt: str, **override_parameters: Any) -> ImageGenerationOutput:
        parameters = self._merge_parameters(**override_parameters)
        logger.debug(f'{prompt=}, {parameters=}')
        return self._image_generation(prompt, parameters)

    async def async_generate(self, prompt: str, **override_parameters: Any) -> ImageGenerationOutput:
        parameters = self._merge_parameters(**override_parameters)
        logger.debug(f'{prompt=}, {parameters=}')
        return await self._async_image_generation(prompt, parameters)
