from __future__ import annotations

from typing import Type

from generate.chat_completion.base import ChatCompletionModel, RemoteChatCompletionModel
from generate.chat_completion.message import (
    AssistantMessage,
    FunctionMessage,
    Messages,
    Prompt,
    SystemMessage,
    ToolMessage,
    UserMessage,
    UserMultiPartMessage,
)
from generate.chat_completion.model_output import ChatCompletionOutput, ChatCompletionStreamOutput
from generate.chat_completion.models import (
    AnthropicChat,
    AnthropicChatParameters,
    AzureChat,
    BaichuanChat,
    BaichuanChatParameters,
    DashScopeChat,
    DashScopeChatParameters,
    DashScopeMultiModalChat,
    DashScopeMultiModalChatParameters,
    DeepSeekChat,
    DeepSeekChatParameters,
    HunyuanChat,
    HunyuanChatParameters,
    MinimaxChat,
    MinimaxChatParameters,
    MinimaxProChat,
    MinimaxProChatParameters,
    MoonshotChat,
    MoonshotChatParameters,
    OpenAIChat,
    OpenAIChatParameters,
    StepFunChat,
    StepFunChatParameters,
    WenxinChat,
    WenxinChatParameters,
    YiChat,
    YiChatParameters,
    ZhipuChat,
    ZhipuChatParameters,
)
from generate.chat_completion.printer import MessagePrinter, SimpleMessagePrinter
from generate.chat_completion.tool import Tool, get_json_schema, tool
from generate.model import ModelParameters

ChatModels: list[tuple[Type[ChatCompletionModel], Type[ModelParameters]]] = [
    (AzureChat, OpenAIChatParameters),
    (AnthropicChat, AnthropicChatParameters),
    (OpenAIChat, OpenAIChatParameters),
    (MinimaxChat, MinimaxChatParameters),
    (MinimaxProChat, MinimaxProChatParameters),
    (ZhipuChat, ZhipuChatParameters),
    (WenxinChat, WenxinChatParameters),
    (HunyuanChat, HunyuanChatParameters),
    (BaichuanChat, BaichuanChatParameters),
    (DashScopeChat, DashScopeChatParameters),
    (DashScopeMultiModalChat, DashScopeMultiModalChatParameters),
    (MoonshotChat, MoonshotChatParameters),
    (DeepSeekChat, DashScopeChatParameters),
    (StepFunChat, StepFunChatParameters),
    (YiChat, YiChatParameters),
]

ChatModelRegistry: dict[str, tuple[Type[ChatCompletionModel], Type[ModelParameters]]] = {
    model_cls.model_type: (model_cls, parameter_cls) for model_cls, parameter_cls in ChatModels
}


__all__ = [
    'ChatCompletionModel',
    'RemoteChatCompletionModel',
    'ChatCompletionOutput',
    'ChatCompletionStreamOutput',
    'ModelParameters',
    'AzureChat',
    'MinimaxChat',
    'MinimaxChatParameters',
    'MinimaxProChat',
    'MinimaxProChatParameters',
    'OpenAIChat',
    'OpenAIChatParameters',
    'ZhipuChat',
    'ZhipuChatParameters',
    'WenxinChat',
    'WenxinChatParameters',
    'HunyuanChat',
    'HunyuanChatParameters',
    'BaichuanChat',
    'BaichuanChatParameters',
    'YiChat',
    'YiChatParameters',
    'StepFunChat',
    'StepFunChatParameters',
    'AnthropicChat',
    'AnthropicChatParameters',
    'DashScopeChat',
    'DashScopeChatParameters',
    'DashScopeMultiModalChat',
    'DashScopeMultiModalChatParameters',
    'MoonshotChat',
    'MoonshotChatParameters',
    'DeepSeekChat',
    'DeepSeekChatParameters',
    'MessagePrinter',
    'SimpleMessagePrinter',
    'get_json_schema',
    'tool',
    'Tool',
    'Prompt',
    'Messages',
    'SystemMessage',
    'UserMessage',
    'AssistantMessage',
    'ToolMessage',
    'FunctionMessage',
    'UserMultiPartMessage',
]
