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
    DeepSeekChat,
    DeepSeekChatParameters,
    MinimaxChat,
    MinimaxChatParameters,
    MoonshotChat,
    MoonshotChatParameters,
    OpenAIChat,
    OpenAIChatParameters,
    OpenRouterChat,
    OpenRouterChatParameters,
    StepFunChat,
    StepFunChatParameters,
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
    (ZhipuChat, ZhipuChatParameters),
    (BaichuanChat, BaichuanChatParameters),
    (DashScopeChat, DashScopeChatParameters),
    (MoonshotChat, MoonshotChatParameters),
    (DeepSeekChat, DashScopeChatParameters),
    (StepFunChat, StepFunChatParameters),
    (YiChat, YiChatParameters),
    (OpenRouterChat, OpenRouterChatParameters),
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
    'OpenAIChat',
    'OpenAIChatParameters',
    'ZhipuChat',
    'ZhipuChatParameters',
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
    'MoonshotChat',
    'MoonshotChatParameters',
    'DeepSeekChat',
    'DeepSeekChatParameters',
    'OpenRouterChat',
    'OpenRouterChatParameters',
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
