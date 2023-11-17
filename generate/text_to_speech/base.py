import logging
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Optional, TypeVar

from generate.model import GenerateModel, ModelOutput, ModelParameters

P = TypeVar('P', bound=ModelParameters)
logger = logging.getLogger(__name__)


class TextToSpeechOutput(ModelOutput):
    audio: bytes
    audio_format: str
    cost: Optional[float] = None


class TextToSpeechModel(GenerateModel[P, str, TextToSpeechOutput], ABC):
    model_task: ClassVar[str] = 'text_to_speech'
    model_type: ClassVar[str]

    def __init__(self, parameters: P) -> None:
        self.parameters = parameters

    @abstractmethod
    def _text_to_speech(self, text: str, parameters: P) -> TextToSpeechOutput:
        ...

    @abstractmethod
    async def _async_text_to_speech(self, text: str, parameters: P) -> TextToSpeechOutput:
        ...

    def generate(self, prompt: str, **override_parameters: Any) -> TextToSpeechOutput:
        parameters = self._merge_parameters(**override_parameters)
        logger.debug(f'{prompt=}, {parameters=}')
        return self._text_to_speech(prompt, parameters)

    async def async_generate(self, prompt: str, **override_parameters: Any) -> TextToSpeechOutput:
        parameters = self._merge_parameters(**override_parameters)
        logger.debug(f'{prompt=}, {parameters=}')
        return await self._async_text_to_speech(prompt, parameters)
