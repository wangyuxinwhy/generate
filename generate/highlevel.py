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


def load_chat_model(model_id: str) -> ChatCompletionModel:
    if '/' not in model_id:
        model_type = model_id
        return ChatModelRegistry[model_type][0]()
    model_type, name = model_id.split('/', maxsplit=1)
    model_cls = ChatModelRegistry[model_type][0]
    return model_cls.from_name(name)


def generate_text(prompt: Prompt, model_id: str = 'openai', **kwargs: Any) -> ChatCompletionOutput:
    model = load_chat_model(model_id)
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
