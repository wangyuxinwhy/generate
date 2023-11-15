from __future__ import annotations

from typing import Any

from generate.chat_completion import ChatCompletionModel, ChatCompletionOutput, ChatModelRegistry
from generate.chat_completion.message import Prompt
from generate.image_generation import ImageGenerationModel, ImageGenerationModelRegistry, ImageGenerationOutput
from generate.text_to_speech import SpeechModelRegistry, TextToSpeechModel, TextToSpeechOutput


def load_chat_model(model_id: str, **kwargs: Any) -> ChatCompletionModel:
    if '/' not in model_id:
        model_type = model_id
        return ChatModelRegistry[model_type][0](**kwargs)  # type: ignore
    model_type, name = model_id.split('/')
    model_cls = ChatModelRegistry[model_type][0]
    return model_cls.from_name(name, **kwargs)


def load_speech_model(speech_model_id: str, **kwargs: Any) -> TextToSpeechModel:
    if '/' not in speech_model_id:
        model_type = speech_model_id
        return SpeechModelRegistry[model_type][0](**kwargs)  # type: ignore
    model_type, name = speech_model_id.split('/')
    model_cls = SpeechModelRegistry[model_type][0]
    return model_cls.from_name(name, **kwargs)


def load_image_generation_model(model_id: str, **kwargs: Any) -> ImageGenerationModel:
    if '/' not in model_id:
        model_type = model_id
        return ImageGenerationModelRegistry[model_type][0](**kwargs)  # type: ignore
    model_type, name = model_id.split('/')
    model_cls = ImageGenerationModelRegistry[model_type][0]
    return model_cls.from_name(name, **kwargs)


def generate_text(prompt: Prompt, model_id: str = 'openai/gpt-3.5-turbo', **kwargs: Any) -> ChatCompletionOutput:
    model = load_chat_model(model_id)
    return model.generate(prompt, **kwargs)


def generate_speech(text: str, model_id: str = 'openai/tts-1', **kwargs: Any) -> TextToSpeechOutput:
    model = load_speech_model(model_id)
    return model.generate(text, **kwargs)


def generate_image(prompt: str, model_id: str = 'openai/dall-e-3', **kwargs: Any) -> ImageGenerationOutput:
    model = load_image_generation_model(model_id)
    return model.generate(prompt, **kwargs)
