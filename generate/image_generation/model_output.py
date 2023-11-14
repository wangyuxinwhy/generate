from typing import List

from generate.model import ModelOutput


class ImageGenerationModelOutput(ModelOutput):
    images: List[bytes]
