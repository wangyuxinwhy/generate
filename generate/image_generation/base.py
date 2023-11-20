import logging
from abc import ABC
from typing import ClassVar, List, Optional

from pydantic import BaseModel

from generate.model import GenerateModel, ModelOutput

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
