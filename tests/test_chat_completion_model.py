from __future__ import annotations

import asyncio
from typing import Any, Type

import pytest

from generate.chat_completion import (
    ChatCompletionModel,
    ChatCompletionStreamOutput,
    ChatModelRegistry,
    ChatModels,
    RemoteChatCompletionModel,
)
from generate.chat_completion.message import Prompt
from generate.chat_completion.models.azure import AzureChat
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
    if issubclass(model_cls, AzureChat):
        return

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
    text = ''.join(output.stream.delta for output in outputs)
    assert text == outputs[-1].reply
    assert outputs[0].stream.control == 'start'
    assert outputs[-1].reply != ''
    assert async_output.reply != ''


@pytest.mark.parametrize(
    ('model_cls'),
    get_pytest_params(
        'test_multimodel_chat_completion',
        ChatModelRegistry,
        types='model_cls',
        include=['dashscope_multimodal', 'zhipu', 'openai', 'anthropic'],
    ),
)
def test_multimodel_chat_completion(model_cls: Type[ChatCompletionModel]) -> None:
    user_message = {
        'role': 'user',
        'content': [
            {'text': '这个图片是哪里？'},
            {'image_url': {'url': 'https://dashscope.oss-cn-beijing.aliyuncs.com/images/dog_and_girl.jpeg'}},
        ],
    }
    if model_cls.model_type == 'openai':
        model = model_cls(model='gpt-4-vision-preview')
    elif model_cls.model_type == 'anthropic':
        model = model_cls(model='claude-3-sonnet-20240229')
    else:
        model = model_cls()
    output = model.generate(user_message)
    assert output.reply != ''


async def async_stream_helper(model: ChatCompletionModel, prompt: Prompt) -> ChatCompletionStreamOutput:
    async for output in model.async_stream_generate(prompt):
        if output.stream.control == 'finish':
            return output
    raise RuntimeError('Stream did not finish')


@pytest.mark.parametrize(
    ('model_cls'),
    get_pytest_params(
        'test_how_to_settings',
        ChatModelRegistry,
        types='model_cls',
        exclude=['azure'],
    ),
)
def test_how_to_settings(model_cls: Type[RemoteChatCompletionModel]) -> None:
    how_to_settings = model_cls.how_to_settings()
    assert model_cls.__name__ in how_to_settings
