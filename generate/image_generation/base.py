import logging
from abc import ABC
from typing import ClassVar, List, Optional, get_type_hints

from pydantic import BaseModel

from generate.http import HttpClient
from generate.model import GenerateModel, ModelOutput, ModelParameters
from generate.platforms.base import PlatformSettings

logger = logging.getLogger(__name__)


class GeneratedImage(BaseModel):
    url: Optional[str] = None
    prompt: str
    image_format: str
    content: bytes


class ImageGenerationOutput(ModelOutput):
    images: List[GeneratedImage] = []


class ImageGenerationModel(GenerateModel[str, ImageGenerationOutput], ABC):
    model_task: ClassVar[str] = 'image_generation'
    model_type: ClassVar[str]


class RemoteImageGenerationModel(ImageGenerationModel):
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
