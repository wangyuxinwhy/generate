from typing import List, Optional

from pydantic import BaseModel

from generate.model import ModelOutput


class GeneratedImage(BaseModel):
    url: Optional[str] = None
    prompt: str
    image_format: str
    content: bytes


class ImageGenerationOutput(ModelOutput):
    images: List[GeneratedImage] = []
