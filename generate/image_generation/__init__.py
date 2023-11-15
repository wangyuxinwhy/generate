from __future__ import annotations

from typing import Type

from generate.image_generation.base import ImageGenerationModel
from generate.image_generation.model_output import GeneratedImage, ImageGenerationOutput
from generate.image_generation.models import OpenAIImageGeneration, OpenAIImageGenerationParameters
from generate.parameters import ModelParameters

ImageGenerationModels: list[tuple[Type[ImageGenerationModel], Type[ModelParameters]]] = [
    (OpenAIImageGeneration, OpenAIImageGenerationParameters),
]

ImageGenerationModelRegistry: dict[str, tuple[Type[ImageGenerationModel], Type[ModelParameters]]] = {
    model_cls.model_type: (model_cls, parameter_cls) for model_cls, parameter_cls in ImageGenerationModels
}

__all__ = [
    'ImageGenerationModel',
    'ImageGenerationOutput',
    'GeneratedImage',
    'OpenAIImageGeneration',
    'OpenAIImageGenerationParameters',
]
