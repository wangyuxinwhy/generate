from __future__ import annotations

from typing import Type

from generate.chat_completion.base import ChatCompletionModel
from generate.chat_completion.function_call import function, get_json_schema
from generate.chat_completion.model_output import ChatCompletionOutput, ChatCompletionStreamOutput
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
from generate.chat_completion.printer import MessagePrinter, SimpleMessagePrinter
from generate.model import ModelParameters

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


__all__ = [
    'ChatCompletionModel',
    'ChatCompletionOutput',
    'ChatCompletionStreamOutput',
    'ModelParameters',
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
    'MessagePrinter',
    'SimpleMessagePrinter',
    'get_json_schema',
    'function',
]
