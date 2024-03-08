from __future__ import annotations

import json
from typing import AsyncIterator, ClassVar, Iterator, List, Literal, Optional

from pydantic import Field, PositiveInt
from typing_extensions import Annotated, TypedDict, Unpack, override

from generate.chat_completion.base import RemoteChatCompletionModel
from generate.chat_completion.message import (
    AssistantMessage,
    Message,
    MessageTypeError,
    Prompt,
    SystemMessage,
    UserMessage,
    ensure_messages,
)
from generate.chat_completion.model_output import ChatCompletionOutput, ChatCompletionStreamOutput
from generate.chat_completion.stream_manager import StreamManager
from generate.http import (
    HttpClient,
    HttpxPostKwargs,
    ResponseValue,
)
from generate.model import ModelParameters, RemoteModelParametersDict
from generate.platforms.dashscope import DashScopeSettings
from generate.types import Probability


class DashscopeMessage(TypedDict):
    role: Literal['user', 'assistant', 'system']
    content: str


class DashScopeChatParameters(ModelParameters):
    seed: Optional[PositiveInt] = None
    max_tokens: Optional[PositiveInt] = None
    top_p: Optional[Probability] = Field(default=None, alias='TopP')
    top_k: Optional[Annotated[int, Field(ge=0, le=100)]] = None
    repetition_penalty: Optional[float] = None
    temperature: Optional[Annotated[float, Field(gt=0, le=2)]] = None
    stop: Optional[List[str]] = None
    search: Annotated[Optional[bool], Field(alias='enable_search')] = None


class DashScopeChatParametersDict(RemoteModelParametersDict, total=False):
    seed: Optional[PositiveInt]
    max_tokens: Optional[PositiveInt]
    top_p: Optional[Probability]
    top_k: Optional[Annotated[int, Field(ge=0, le=100)]]
    repetition_penalty: Optional[float]
    temperature: Optional[Annotated[float, Field(gt=0, le=2)]]
    stop: Optional[List[str]]
    search: Optional[bool]


class DashScopeChat(RemoteChatCompletionModel):
    model_type: ClassVar[str] = 'dashscope'
    available_models: ClassVar[List[str]] = ['qwen-turbo', 'qwen-plus', 'qwen-max', 'qwen-max-1201', 'qwen-max-longcontext']

    parameters: DashScopeChatParameters
    settings: DashScopeSettings

    def __init__(
        self,
        model: str = 'qwen-max',
        parameters: DashScopeChatParameters | None = None,
        settings: DashScopeSettings | None = None,
        http_client: HttpClient | None = None,
    ) -> None:
        parameters = parameters or DashScopeChatParameters()
        settings = settings or DashScopeSettings()  # type: ignore
        http_client = http_client or HttpClient()
        super().__init__(model=model, parameters=parameters, settings=settings, http_client=http_client)

    @override
    def generate(self, prompt: Prompt, **kwargs: Unpack[DashScopeChatParametersDict]) -> ChatCompletionOutput:
        return super().generate(prompt, **kwargs)

    @override
    async def async_generate(self, prompt: Prompt, **kwargs: Unpack[DashScopeChatParametersDict]) -> ChatCompletionOutput:
        return await super().async_generate(prompt, **kwargs)

    @override
    def stream_generate(
        self, prompt: Prompt, **kwargs: Unpack[DashScopeChatParametersDict]
    ) -> Iterator[ChatCompletionStreamOutput]:
        yield from super().stream_generate(prompt, **kwargs)

    @override
    async def async_stream_generate(
        self, prompt: Prompt, **kwargs: Unpack[DashScopeChatParametersDict]
    ) -> AsyncIterator[ChatCompletionStreamOutput]:
        async for output in super().async_stream_generate(prompt, **kwargs):
            yield output

    @override
    def _get_request_parameters(
        self, prompt: Prompt, stream: bool = False, **kwargs: Unpack[DashScopeChatParametersDict]
    ) -> HttpxPostKwargs:
        messages = ensure_messages(prompt)
        parameters = self.parameters.clone_with_changes(**kwargs)
        zhipu_messages = [self._convert_message(message) for message in messages]
        headers = {
            'Authorization': self.settings.api_key.get_secret_value(),
            'Content-Type': 'application/json',
        }
        if stream:
            headers['Accept'] = 'text/event-stream'

        params = {
            'input': {
                'messages': zhipu_messages,
            },
            'model': self.model,
            'parameters': parameters.custom_model_dump(),
        }
        return {
            'url': self.settings.api_base + '/services/aigc/text-generation/generation',
            'headers': headers,
            'json': params,
        }

    @override
    def _process_reponse(self, response: ResponseValue) -> ChatCompletionOutput:
        return ChatCompletionOutput(
            model_info=self.model_info,
            message=AssistantMessage(content=response['output']['text']),
            cost=self._calculate_cost(response['usage']['total_tokens']),
            extra={'usage': response['usage'], 'request_id': response['request_id']},
            finish_reason=response['output']['finish_reason'],
        )

    @override
    def _process_stream_line(self, line: str, stream_manager: StreamManager) -> ChatCompletionStreamOutput | None:
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            return None

        finish_reason = data['output']['finish_reason'] or None
        reply = data['output']['text']
        stream_manager.extra['usage'] = data['usage']
        stream_manager.extra['request_id'] = data['request_id']
        if finish_reason == 'stop':
            stream_manager.finish_reason = finish_reason
            stream_manager.delta = ''
            stream_manager.cost = self._calculate_cost(total_tokens=stream_manager.extra['usage']['total_tokens'])
            return stream_manager.build_stream_output()
        stream_manager.delta = reply[len(stream_manager.content) :]
        return stream_manager.build_stream_output()

    @staticmethod
    def _convert_message(message: Message) -> DashscopeMessage:
        if isinstance(message, UserMessage):
            return {'role': 'user', 'content': message.content}
        if isinstance(message, AssistantMessage):
            return {'role': 'assistant', 'content': message.content}
        if isinstance(message, SystemMessage):
            return {'role': 'system', 'content': message.content}
        raise MessageTypeError(message, (UserMessage, AssistantMessage, SystemMessage))

    def _calculate_cost(self, total_tokens: int) -> Optional[float]:
        if self.model == 'qwen-turbo':
            return total_tokens * 0.008 / 1000
        if self.model == 'qwen-plus':
            return total_tokens * 0.04 / 1000
        return None
