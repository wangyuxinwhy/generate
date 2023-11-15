from __future__ import annotations

from typing import Any, Type

from generate.parameters import ModelParameters
from generate.text_to_speech.base import TextToSpeechModel
from generate.text_to_speech.model_output import TextToSpeechOutput
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


def load_speech_model(speech_model_id: str, **kwargs: Any) -> TextToSpeechModel:
    if '/' not in speech_model_id:
        model_type = speech_model_id
        return SpeechModelRegistry[model_type][0](**kwargs)  # type: ignore
    model_type, name = speech_model_id.split('/')
    model_cls = SpeechModelRegistry[model_type][0]
    return model_cls.from_name(name, **kwargs)


def list_speech_model_types() -> list[str]:
    return list(SpeechModelRegistry.keys())


def generate_speech(text: str, model_id: str = 'openai/tts-1', **kwargs: Any) -> TextToSpeechOutput:
    model = load_speech_model(model_id, **kwargs)
    return model.generate(text, **kwargs)


__all__ = [
    'load_speech_model',
    'list_speech_model_types',
    'generate_speech',
    'TextToSpeechModel',
    'TextToSpeechOutput',
    'MinimaxSpeech',
    'MinimaxProSpeech',
    'MinimaxSpeechParameters',
    'MinimaxProSpeechParameters',
    'OpenAISpeech',
    'OpenAISpeechParameters',
]
