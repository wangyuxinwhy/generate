import logging
from abc import ABC
from typing import ClassVar, Optional, TypeVar

from generate.model import GenerateModel, ModelOutput, ModelParameters

P = TypeVar('P', bound=ModelParameters)
logger = logging.getLogger(__name__)


class TextToSpeechOutput(ModelOutput):
    audio: bytes
    audio_format: str
    cost: Optional[float] = None


class TextToSpeechModel(GenerateModel[str, TextToSpeechOutput], ABC):
    model_task: ClassVar[str] = 'text_to_speech'
