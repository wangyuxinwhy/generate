from __future__ import annotations

from typing import Any, Type

from generate.chat_completion.base import ChatCompletionModel
from generate.chat_completion.http_chat import HttpChatModel, HttpModelInitKwargs
from generate.chat_completion.model_output import ChatCompletionModelOutput, ChatCompletionModelStreamOutput
from generate.chat_completion.models import (
    AzureChat,
    BaichuanChat,
    BaichuanChatParameters,
    BailianChat,
    BailianChatParameters,
    HunyuanChat,
    HunyuanChatParameters,
    MinimaxChat,
    MinimaxChatParameters,
    MinimaxProChat,
    MinimaxProChatParameters,
    OpenAIChat,
    OpenAIChatParameters,
    WenxinChat,
    WenxinChatParameters,
    ZhipuCharacterChat,
    ZhipuCharacterChatParameters,
    ZhipuChat,
    ZhipuChatParameters,
)
from generate.parameters import ModelParameters

ChatModels: list[tuple[Type[ChatCompletionModel], Type[ModelParameters]]] = [
    (AzureChat, OpenAIChatParameters),
    (OpenAIChat, OpenAIChatParameters),
    (MinimaxProChat, MinimaxProChatParameters),
    (MinimaxChat, MinimaxProChatParameters),
    (ZhipuChat, ZhipuChatParameters),
    (ZhipuCharacterChat, ZhipuCharacterChatParameters),
    (WenxinChat, WenxinChatParameters),
    (HunyuanChat, HunyuanChatParameters),
    (BaichuanChat, BaichuanChatParameters),
    (BailianChat, BailianChatParameters),
]

ChatModelRegistry: dict[str, tuple[Type[ChatCompletionModel], Type[ModelParameters]]] = {
    model_cls.model_type: (model_cls, parameter_cls) for model_cls, parameter_cls in ChatModels
}


def load_chat_model(model_id: str, **kwargs: Any) -> ChatCompletionModel:
    if '/' not in model_id:
        model_type = model_id
        return ChatModelRegistry[model_type][0](**kwargs)  # type: ignore
    model_type, name = model_id.split('/')
    model_cls = ChatModelRegistry[model_type][0]
    return model_cls.from_name(name, **kwargs)


def list_chat_model_types() -> list[str]:
    return list(ChatModelRegistry.keys())


__all__ = [
    'ChatCompletionModel',
    'ChatCompletionModelOutput',
    'ChatCompletionModelStreamOutput',
    'ModelParameters',
    'HttpChatModel',
    'HttpModelInitKwargs',
    'AzureChat',
    'MinimaxProChat',
    'MinimaxProChatParameters',
    'MinimaxChat',
    'MinimaxChatParameters',
    'OpenAIChat',
    'OpenAIChatParameters',
    'ZhipuChat',
    'ZhipuChatParameters',
    'ZhipuCharacterChat',
    'ZhipuCharacterChatParameters',
    'WenxinChat',
    'WenxinChatParameters',
    'HunyuanChat',
    'HunyuanChatParameters',
    'BaichuanChat',
    'BaichuanChatParameters',
    'BailianChat',
    'BailianChatParameters',
]
