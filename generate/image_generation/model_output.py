from typing import List

from generate.model import ModelOutput


class ImageGenerationOutput(ModelOutput):
    images: List[bytes] = []
