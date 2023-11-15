import logging
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Generic, TypeVar

from typing_extensions import Self

from generate.model import ModelInfo
from generate.parameters import ModelParameters
from generate.text_to_speech.model_output import TextToSpeechOutput

P = TypeVar('P', bound=ModelParameters)
logger = logging.getLogger(__name__)


class TextToSpeechModel(Generic[P], ABC):
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
    def _text_to_speech(self, text: str, parameters: P) -> TextToSpeechOutput:
        ...

    @abstractmethod
    async def _async_text_to_speech(self, text: str, parameters: P) -> TextToSpeechOutput:
        ...

    @property
    def model_info(self) -> ModelInfo:
        return ModelInfo(task='text_to_speech', type=self.model_type, name=self.name)

    def generate(self, text: str, **override_parameters: Any) -> TextToSpeechOutput:
        parameters = self._merge_parameters(**override_parameters)
        logger.debug(f'{text=}, {parameters=}')
        return self._text_to_speech(text, parameters)

    async def async_generate(self, text: str, **override_parameters: Any) -> TextToSpeechOutput:
        parameters = self._merge_parameters(**override_parameters)
        logger.debug(f'{text=}, {parameters=}')
        return await self._async_text_to_speech(text, parameters)

    def _merge_parameters(self, **override_parameters: Any) -> P:
        return self.parameters.__class__.model_validate(
            {**self.parameters.model_dump(exclude_unset=True), **override_parameters}
        )
