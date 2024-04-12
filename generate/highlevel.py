from __future__ import annotations

import asyncio
from typing import Any, Literal, Sequence

from typing_extensions import overload

from generate.chat_completion import (
    ChatCompletionModel,
    ChatCompletionOutput,
    ChatModelRegistry,
    Prompt,
)
from generate.image_generation import (
    ImageGenerationModel,
    ImageGenerationModelRegistry,
    ImageGenerationOutput,
)
from generate.text_to_speech import (
    SpeechModelRegistry,
    TextToSpeechModel,
    TextToSpeechOutput,
)


def load_chat_model(model_id: str) -> ChatCompletionModel:
    if '/' not in model_id:
        model_type = model_id
        return ChatModelRegistry[model_type][0]()
    model_type, name = model_id.split('/', maxsplit=1)
    model_cls = ChatModelRegistry[model_type][0]
    return model_cls.from_name(name)


def load_speech_model(model_id: str) -> TextToSpeechModel:
    if '/' not in model_id:
        model_type = model_id
        return SpeechModelRegistry[model_type][0]()
    model_type, name = model_id.split('/', maxsplit=1)
    model_cls = SpeechModelRegistry[model_type][0]
    return model_cls.from_name(name)


def load_image_generation_model(model_id: str) -> ImageGenerationModel:
    if '/' not in model_id:
        model_type = model_id
        return ImageGenerationModelRegistry[model_type][0]()
    model_type, name = model_id.split('/', maxsplit=1)
    model_cls = ImageGenerationModelRegistry[model_type][0]
    return model_cls.from_name(name)


def generate_text(prompt: Prompt, model_id: str = 'openai', **kwargs: Any) -> ChatCompletionOutput:
    model = load_chat_model(model_id)
    return model.generate(prompt, **kwargs)


def generate_speech(text: str, model_id: str = 'openai', **kwargs: Any) -> TextToSpeechOutput:
    model = load_speech_model(model_id)
    return model.generate(text, **kwargs)


def generate_image(prompt: str, model_id: str = 'openai', **kwargs: Any) -> ImageGenerationOutput:
    model = load_image_generation_model(model_id)
    return model.generate(prompt, **kwargs)


@overload
async def multimodel_generate_text(
    prompt: Prompt, model_ids: Sequence[str], ignore_error: Literal[False] = False, **kwargs: Any
) -> Sequence[ChatCompletionOutput]:
    ...


@overload
async def multimodel_generate_text(
    prompt: Prompt, model_ids: Sequence[str], ignore_error: Literal[True] = True, **kwargs: Any
) -> Sequence[ChatCompletionOutput | None]:
    ...


async def multimodel_generate_text(
    prompt: Prompt, model_ids: Sequence[str], ignore_error: bool = False, **kwargs: Any
) -> Sequence[ChatCompletionOutput | None]:
    async def _generate_text(prompt: Prompt, model_id: str, ignore_error: bool = False) -> ChatCompletionOutput | None:
        try:
            model = load_chat_model(model_id)
            return await model.async_generate(prompt, **kwargs)
        except Exception:
            if ignore_error:
                return None
            raise

    return await asyncio.gather(*[_generate_text(prompt, model_id, ignore_error) for model_id in model_ids])
