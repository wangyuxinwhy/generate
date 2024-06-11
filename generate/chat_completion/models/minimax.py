from __future__ import annotations

import json
from typing import Any, AsyncIterator, ClassVar, Dict, Iterator, List, Optional

from pydantic import PositiveInt, field_validator
from typing_extensions import Unpack, override

from generate.chat_completion.message import (
    Prompt,
)
from generate.chat_completion.message.converter import MessageConverter
from generate.chat_completion.message.core import Messages
from generate.chat_completion.model_output import ChatCompletionOutput, ChatCompletionStreamOutput, FinishReason, Usage
from generate.chat_completion.models.openai_like import OpenAILikeChat, OpenAITool
from generate.chat_completion.tool import SupportToolCall
from generate.http import (
    HttpClient,
    HttpxPostKwargs,
)
from generate.model import ModelParameters, RemoteModelParametersDict
from generate.platforms.minimax import MinimaxSettings
from generate.types import Probability, Temperature


class MinimaxChatParameters(ModelParameters):
    temperature: Optional[Temperature] = None
    top_p: Optional[Probability] = None
    max_tokens: Optional[PositiveInt] = None
    tool_choice: Optional[str] = None
    tools: Optional[List[OpenAITool]] = None

    @field_validator('temperature', 'top_p')
    @classmethod
    def can_not_equal_zero(cls, value: Optional[Temperature]) -> Optional[Temperature]:
        if value == 0:
            return 0.01
        return value


class MinimaxChatParametersDict(RemoteModelParametersDict, total=False):
    temperature: Optional[Temperature]
    top_p: Optional[Probability]
    max_tokens: Optional[PositiveInt]
    tool_choice: Optional[str]
    tools: Optional[List[OpenAITool]]


class MinimaxChat(OpenAILikeChat, SupportToolCall):
    model_type: ClassVar[str] = 'minimax'
    available_models: ClassVar[List[str]] = ['abab5.5-chat', 'abab5.5s-chat', 'abab6-chat', 'abab6.5-chat']
    CHAT_COMPLETION_ENDPOINT: ClassVar[str] = '/text/chatcompletion_v2'

    parameters: MinimaxChatParameters
    settings: MinimaxSettings

    def __init__(
        self,
        model: str = 'abab5.5-chat',
        parameters: MinimaxChatParameters | None = None,
        settings: MinimaxSettings | None = None,
        http_client: HttpClient | None = None,
        message_converter: MessageConverter | None = None,
    ) -> None:
        parameters = parameters or MinimaxChatParameters()
        settings = settings or MinimaxSettings()  # type: ignore
        http_client = http_client or HttpClient()
        super().__init__(
            model=model,
            parameters=parameters,
            settings=settings,
            http_client=http_client,
            message_converter=message_converter,
        )

    @override
    def _get_request_parameters(self, messages: Messages, stream: bool = False, **kwargs: Any) -> HttpxPostKwargs:
        http_kwargs = super()._get_request_parameters(messages, stream, **kwargs)
        http_kwargs['url'] = self.settings.api_base + self.CHAT_COMPLETION_ENDPOINT
        if 'tools' in http_kwargs['json']:
            # Serialize jsonschema dict to JSON string for Minimax compatibility
            for tool in http_kwargs['json']['tools']:
                if 'function' in tool:
                    tool['function']['parameters'] = json.dumps(tool['function']['parameters'])
            if http_kwargs['json'].get('tool_choice', None):
                http_kwargs['json']['tool_choice'] = 'auto'
        return http_kwargs

    @override
    def _parse_finish_reason(self, response: Dict[str, Any]) -> FinishReason | None:
        choice = response['choices'][0]
        if 'finish_reason' in choice and 'delta' not in choice:
            return FinishReason(choice['finish_reason'])
        return None

    @override
    def _parse_usage(self, response: dict[str, Any]) -> Usage:
        if usage := response.get('usage'):
            return Usage(input_tokens=0, output_tokens=usage['total_tokens'])
        return Usage()

    @override
    def generate(self, prompt: Prompt, **kwargs: Unpack[MinimaxChatParametersDict]) -> ChatCompletionOutput:
        return super().generate(prompt, **kwargs)  # type: ignore

    @override
    async def async_generate(self, prompt: Prompt, **kwargs: Unpack[MinimaxChatParametersDict]) -> ChatCompletionOutput:
        return await super().async_generate(prompt, **kwargs)  # type: ignore

    @override
    def stream_generate(
        self, prompt: Prompt, **kwargs: Unpack[MinimaxChatParametersDict]
    ) -> Iterator[ChatCompletionStreamOutput]:
        yield from super().stream_generate(prompt, **kwargs)  # type: ignore

    @override
    async def async_stream_generate(
        self, prompt: Prompt, **kwargs: Unpack[MinimaxChatParametersDict]
    ) -> AsyncIterator[ChatCompletionStreamOutput]:
        async for stream_output in super().async_stream_generate(prompt, **kwargs):  # type: ignore
            yield stream_output
