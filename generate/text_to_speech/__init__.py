from __future__ import annotations

from typing import Type

from generate.model import ModelParameters
from generate.text_to_speech.base import TextToSpeechModel, TextToSpeechOutput
from generate.text_to_speech.models import (
    MinimaxProSpeech,
    MinimaxProSpeechParameters,
    MinimaxSpeech,
    MinimaxSpeechParameters,
    OpenAISpeech,
    OpenAISpeechParameters,
)

SpeechModels: list[tuple[Type[TextToSpeechModel], Type[ModelParameters]]] = [
    (MinimaxSpeech, MinimaxSpeechParameters),
    (MinimaxProSpeech, MinimaxProSpeechParameters),
    (OpenAISpeech, OpenAISpeechParameters),
]

SpeechModelRegistry: dict[str, tuple[Type[TextToSpeechModel], Type[ModelParameters]]] = {
    model_cls.model_type: (model_cls, parameter_cls) for model_cls, parameter_cls in SpeechModels
}


__all__ = [
    'TextToSpeechModel',
    'TextToSpeechOutput',
    'MinimaxSpeech',
    'MinimaxProSpeech',
    'MinimaxSpeechParameters',
    'MinimaxProSpeechParameters',
    'OpenAISpeech',
    'OpenAISpeechParameters',
]
