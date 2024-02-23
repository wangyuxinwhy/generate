import logging
from abc import ABC
from typing import ClassVar, Optional, TypeVar, get_type_hints

from generate.http import HttpClient
from generate.model import GenerateModel, ModelOutput, ModelParameters
from generate.platforms.base import PlatformSettings

P = TypeVar('P', bound=ModelParameters)
logger = logging.getLogger(__name__)


class TextToSpeechOutput(ModelOutput):
    audio: bytes
    audio_format: str
    cost: Optional[float] = None


class TextToSpeechModel(GenerateModel[str, TextToSpeechOutput], ABC):
    model_task: ClassVar[str] = 'text_to_speech'


class RemoteTextToSpeechModel(TextToSpeechModel):
    settings: PlatformSettings
    http_client: HttpClient

    def __init__(
        self,
        parameters: ModelParameters,
        settings: PlatformSettings,
        http_client: HttpClient,
    ) -> None:
        self.parameters = parameters
        self.settings = settings
        self.http_client = http_client

    @classmethod
    def how_to_settings(cls) -> str:
        return f'{cls.__name__} Settings\n\n' + get_type_hints(cls)['settings'].how_to_settings()
