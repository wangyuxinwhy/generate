from __future__ import annotations

from typing import Type, TypeVar

from generate.image_generation.base import GeneratedImage, ImageGenerationModel, ImageGenerationOutput
from generate.image_generation.models import (
    BaiduImageGeneration,
    BaiduImageGenerationParameters,
    OpenAIImageGeneration,
    OpenAIImageGenerationParameters,
    QianfanImageGeneration,
    QianfanImageGenerationParameters,
    ZhipuImageGeneration,
)
from generate.model import ModelParameters

P = TypeVar('P', bound=ModelParameters)

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
    'BaiduImageGeneration',
    'BaiduImageGenerationParameters',
    'QianfanImageGeneration',
    'QianfanImageGenerationParameters',
    'ZhipuImageGeneration',
]
