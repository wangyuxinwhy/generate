from __future__ import annotations

import json
import uuid
from typing import Any, AsyncIterator, ClassVar, Dict, Iterator, List, Optional

from pydantic import PositiveInt, field_validator
from typing_extensions import Unpack, override

from generate.chat_completion.message import (
    Prompt,
)
from generate.chat_completion.message.core import AssistantMessage, FunctionMessage, Messages, ToolCall, ToolMessage
from generate.chat_completion.model_output import ChatCompletionOutput, ChatCompletionStreamOutput
from generate.chat_completion.models.openai_like import OpenAILikeChat, OpenAIMessage, OpenAITool
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
    def can_not_equal_zero(cls, v: Optional[Temperature]) -> Optional[Temperature]:
        if v == 0:
            return 0.01
        return v


class MinimaxChatParametersDict(RemoteModelParametersDict, total=False):
    temperature: Optional[Temperature]
    top_p: Optional[Probability]
    max_tokens: Optional[PositiveInt]
    tool_choice: Optional[str]
    tools: Optional[List[OpenAITool]]


class MinimaxChat(OpenAILikeChat):
    model_type: ClassVar[str] = 'minimax'
    avaliable_models: ClassVar[List[str]] = ['abab5.5-chat', 'abab6-chat']

    parameters: MinimaxChatParameters
    settings: MinimaxSettings

    def __init__(
        self,
        model: str = 'abab5.5-chat',
        parameters: MinimaxChatParameters | None = None,
        settings: MinimaxSettings | None = None,
        http_client: HttpClient | None = None,
    ) -> None:
        parameters = parameters or MinimaxChatParameters()
        settings = settings or MinimaxSettings()  # type: ignore
        http_client = http_client or HttpClient()
        model = model
        super().__init__(model=model, parameters=parameters, settings=settings, http_client=http_client)

    @override
    def _get_request_parameters(self, prompt: Prompt, stream: bool = False, **kwargs: Any) -> HttpxPostKwargs:
        http_kwargs = super()._get_request_parameters(prompt, stream, **kwargs)
        http_kwargs['url'] = self.settings.api_base + '/text/chatcompletion_v2'
        if 'tools' in http_kwargs['json']:
            for tool in http_kwargs['json']['tools']:
                if 'function' in tool:
                    tool['function']['parameters'] = json.dumps(tool['function']['parameters'])
            if http_kwargs['json'].get('tool_choice', None):
                http_kwargs['json']['tool_choice'] = 'auto'
        return http_kwargs

    @override
    def _determine_finish_reason(self, response: Dict[str, Any]) -> str | None:
        choice = response['choices'][0]
        if 'finish_reason' in choice and 'delta' not in choice:
            return choice['finish_reason']
        return None

    @override
    def _convert_to_openai_messages(self, messages: Messages) -> List[OpenAIMessage]:
        new_messages = []
        fake_tool_call_id = f'call_{uuid.uuid4()}'
        for message in messages:
            if isinstance(message, AssistantMessage):
                if message.function_call is not None:
                    tool_call = ToolCall(
                        id=fake_tool_call_id,
                        function=message.function_call,
                    )
                    message.tool_calls = [tool_call]
                    message.function_call = None
            elif isinstance(message, FunctionMessage):
                tool_message = ToolMessage(
                    name=message.name,
                    content=message.content,
                    tool_call_id=fake_tool_call_id,
                )
                fake_tool_call_id = f'call_{uuid.uuid4()}'
                new_messages.append(tool_message)
                continue
            new_messages.append(message.model_copy(deep=True))
        return super()._convert_to_openai_messages(new_messages)

    @override
    def generate(self, prompt: Prompt, **kwargs: Unpack[MinimaxChatParametersDict]) -> ChatCompletionOutput:
        return super().generate(prompt, **kwargs)

    @override
    async def async_generate(self, prompt: Prompt, **kwargs: Unpack[MinimaxChatParametersDict]) -> ChatCompletionOutput:
        return await super().async_generate(prompt, **kwargs)

    @override
    def stream_generate(
        self, prompt: Prompt, **kwargs: Unpack[MinimaxChatParametersDict]
    ) -> Iterator[ChatCompletionStreamOutput]:
        yield from super().stream_generate(prompt, **kwargs)

    @override
    async def async_stream_generate(
        self, prompt: Prompt, **kwargs: Unpack[MinimaxChatParametersDict]
    ) -> AsyncIterator[ChatCompletionStreamOutput]:
        async for i in super().async_stream_generate(prompt, **kwargs):
            yield i
