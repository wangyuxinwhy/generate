from __future__ import annotations

import asyncio
from typing import Any, Type

import pytest

from generate.chat_completion import (
    ChatCompletionModel,
    ChatCompletionStreamOutput,
    ChatModelRegistry,
    ChatModels,
)
from generate.chat_completion.message import Prompt
from generate.test import get_pytest_params


def test_model_type_is_unique() -> None:
    assert len(ChatModels) == len(ChatModelRegistry)


@pytest.mark.parametrize(
    ('model_cls'),
    get_pytest_params('test_chat_completion', ChatModelRegistry, types='model_cls'),
)
@pytest.mark.parametrize(
    'parameters',
    [
        {},
        {'temperature': 0.5, 'top_p': 0.85, 'max_tokens': 20},
    ],
)
def test_http_chat_model(model_cls: Type[ChatCompletionModel], parameters: dict[str, Any]) -> None:
    model = model_cls()
    prompt = '这是测试，只回复你好'
    sync_output = model.generate(prompt, **parameters)
    async_output = asyncio.run(model.async_generate(prompt))

    assert sync_output.reply != ''
    assert async_output.reply != ''


@pytest.mark.parametrize(
    'chat_completion_model',
    get_pytest_params('test_stream_chat_completion', ChatModelRegistry, types='model', exclude=['azure']),
)
def test_http_stream_chat_model(chat_completion_model: ChatCompletionModel) -> None:
    prompt = '这是测试，只回复你好'
    outputs = list(chat_completion_model.stream_generate(prompt))
    async_output = asyncio.run(async_stream_helper(chat_completion_model, prompt))

    assert outputs[-1].stream.control == 'finish'
    for output in outputs[1:-1]:
        assert output.stream.control == 'continue'
    assert outputs[0].stream.control == 'start'
    assert outputs[-1].reply != ''
    assert async_output.reply != ''


async def async_stream_helper(model: ChatCompletionModel, prompt: Prompt) -> ChatCompletionStreamOutput:
    async for output in model.async_stream_generate(prompt):
        if output.stream.control == 'finish':
            return output
    raise RuntimeError('Stream did not finish')
