import logging
from abc import ABC
from typing import ClassVar, Optional, TypeVar

from pydantic_settings import BaseSettings

from generate.http import HttpClient
from generate.model import GenerateModel, ModelOutput, ModelParameters

P = TypeVar('P', bound=ModelParameters)
logger = logging.getLogger(__name__)


class TextToSpeechOutput(ModelOutput):
    audio: bytes
    audio_format: str
    cost: Optional[float] = None


class TextToSpeechModel(GenerateModel[str, TextToSpeechOutput], ABC):
    model_task: ClassVar[str] = 'text_to_speech'


class RemoteTextToSpeechModel(TextToSpeechModel):
    def __init__(
        self,
        parameters: ModelParameters,
        settings: BaseSettings,
        http_client: HttpClient,
    ) -> None:
        self.parameters = parameters
        self.settings = settings
        self.http_client = http_client
