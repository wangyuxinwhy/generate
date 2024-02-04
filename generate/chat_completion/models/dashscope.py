from __future__ import annotations

import json
from typing import AsyncIterator, ClassVar, Iterator, List, Literal, Optional

from pydantic import Field, PositiveInt
from typing_extensions import Annotated, Self, TypedDict, Unpack, override

from generate.chat_completion.base import RemoteChatCompletionModel
from generate.chat_completion.message import (
    AssistantMessage,
    Message,
    Messages,
    MessageTypeError,
    Prompt,
    SystemMessage,
    UserMessage,
    UserMultiPartMessage,
    ensure_messages,
)
from generate.chat_completion.message.core import ImageUrlPart, TextPart
from generate.chat_completion.model_output import ChatCompletionOutput, ChatCompletionStreamOutput, Stream
from generate.http import (
    HttpClient,
    HttpxPostKwargs,
    ResponseValue,
)
from generate.model import ModelParameters, ModelParametersDict
from generate.platforms.dashscope import DashScopeSettings
from generate.types import Probability


class DashScopeChatParameters(ModelParameters):
    seed: Optional[PositiveInt] = None
    max_tokens: Optional[PositiveInt] = None
    top_p: Optional[Probability] = Field(default=None, alias='TopP')
    top_k: Optional[Annotated[int, Field(ge=0, le=100)]] = None
    repetition_penalty: Optional[float] = None
    temperature: Optional[Annotated[float, Field(gt=0, le=2)]] = None
    stop: Optional[List[str]] = None
    enable_search: Optional[bool] = None


class DashScopeChatParametersDict(ModelParametersDict):
    seed: int
    max_tokens: int
    top_p: float
    top_k: int
    repetition_penalty: float
    temperature: float
    stop: List[str]
    enable_search: bool


def _convert_message_to_chat_message(message: Message) -> dict[str, str]:
    if isinstance(message, UserMessage):
        return {'role': 'user', 'content': message.content}
    if isinstance(message, AssistantMessage):
        return {'role': 'assistant', 'content': message.content}
    if isinstance(message, SystemMessage):
        return {'role': 'system', 'content': message.content}
    raise MessageTypeError(message, (UserMessage, AssistantMessage, SystemMessage))


def _calculate_cost(model_name: str, total_tokens: int) -> Optional[float]:
    return None


class DashScopeChat(RemoteChatCompletionModel):
    model_type: ClassVar[str] = 'dashscope'

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
        super().__init__(parameters=parameters, settings=settings, http_client=http_client)

        self.model = model

    def _get_request_parameters(self, messages: Messages, parameters: DashScopeChatParameters) -> HttpxPostKwargs:
        zhipu_messages = [_convert_message_to_chat_message(message) for message in messages]
        headers = {
            'Authorization': self.settings.api_key.get_secret_value(),
            'Content-Type': 'application/json',
        }
        params = {
            'input': {
                'messages': zhipu_messages,
            },
            'model': self.model,
            'parameters': parameters.custom_model_dump(),
        }
        return {
            'url': f'{self.settings.api_base}/text-generation/generation',
            'headers': headers,
            'json': params,
        }

    @override
    def generate(self, prompt: Prompt, **kwargs: Unpack[DashScopeChatParametersDict]) -> ChatCompletionOutput:
        messages = ensure_messages(prompt)
        parameters = self.parameters.clone_with_changes(**kwargs)
        request_parameters = self._get_request_parameters(messages, parameters)
        response = self.http_client.post(request_parameters=request_parameters)
        return self._parse_reponse(response.json())

    @override
    async def async_generate(self, prompt: Prompt, **kwargs: Unpack[DashScopeChatParametersDict]) -> ChatCompletionOutput:
        messages = ensure_messages(prompt)
        parameters = self.parameters.clone_with_changes(**kwargs)
        request_parameters = self._get_request_parameters(messages, parameters)
        response = await self.http_client.async_post(request_parameters=request_parameters)
        return self._parse_reponse(response.json())

    def _parse_reponse(self, response: ResponseValue) -> ChatCompletionOutput:
        return ChatCompletionOutput(
            model_info=self.model_info,
            message=AssistantMessage(content=response['output']['text']),
            cost=_calculate_cost(self.model, response['usage']['total_tokens']),
            extra={'usage': response['usage'], 'request_id': response['request_id']},
        )

    def _get_stream_request_parameters(self, messages: Messages, parameters: DashScopeChatParameters) -> HttpxPostKwargs:
        http_parameters = self._get_request_parameters(messages, parameters)
        http_parameters['headers']['Accept'] = 'text/event-stream'
        return http_parameters

    @override
    def stream_generate(
        self, prompt: Prompt, **kwargs: Unpack[DashScopeChatParametersDict]
    ) -> Iterator[ChatCompletionStreamOutput]:
        messages = ensure_messages(prompt)
        parameters = self.parameters.clone_with_changes(**kwargs)
        request_parameters = self._get_stream_request_parameters(messages, parameters)
        message = AssistantMessage(content='')
        is_start = True
        is_finish = False
        for line in self.http_client.stream_post(request_parameters=request_parameters):
            if is_finish:
                continue

            output = self._parse_stream_line(line, message, is_start)
            is_start = False
            is_finish = output.is_finish
            yield output

    @override
    async def async_stream_generate(
        self, prompt: Prompt, **kwargs: Unpack[DashScopeChatParametersDict]
    ) -> AsyncIterator[ChatCompletionStreamOutput]:
        messages = ensure_messages(prompt)
        parameters = self.parameters.clone_with_changes(**kwargs)
        request_parameters = self._get_stream_request_parameters(messages, parameters)
        message = AssistantMessage(content='')
        is_start = True
        is_finish = False
        async for line in self.http_client.async_stream_post(request_parameters=request_parameters):
            if is_finish:
                continue

            output = self._parse_stream_line(line, message, is_start)
            is_start = False
            is_finish = output.is_finish
            yield output

    def _parse_stream_line(self, line: str, message: AssistantMessage, is_start: bool) -> ChatCompletionStreamOutput:
        parsed_line = json.loads(line)
        finish_reason = parsed_line['output']['finish_reason']
        reply = parsed_line['output']['text']
        usage = parsed_line['usage']
        request_id = parsed_line['request_id']
        extra = {
            'usage': usage,
            'response_id': request_id,
        }
        if finish_reason == 'stop':
            return ChatCompletionStreamOutput(
                model_info=self.model_info,
                cost=_calculate_cost(self.model, total_tokens=usage['total_tokens']),
                message=AssistantMessage(content=reply),
                extra=extra,
                finish_reason='stop',
                stream=Stream(delta='', control='finish'),
            )
        delta = reply[len(message.content) :]
        message.content = reply
        return ChatCompletionStreamOutput(
            model_info=self.model_info,
            message=message,
            extra=extra,
            stream=Stream(delta=delta, control='start' if is_start else 'continue'),
        )

    @property
    def name(self) -> str:
        return self.model

    @classmethod
    def from_name(cls, name: str) -> Self:
        return cls(model=name)


class DashScopeMultiModalChatParameters(ModelParameters):
    seed: Optional[PositiveInt] = None
    top_p: Optional[Probability] = Field(default=None, alias='TopP')
    top_k: Optional[Annotated[int, Field(ge=0, le=100)]] = None


class DashScopeMultiModalChatParametersDict(ModelParametersDict, total=False):
    seed: int
    top_p: float
    top_k: int


class DashScopeMultiModalMessage(TypedDict):
    role: str
    content: list[dict[Literal['image', 'text'], str]]


def _convert_message_to_multimodal_chat_message(message: Message) -> DashScopeMultiModalMessage:
    if isinstance(message, UserMessage):
        return {'role': 'user', 'content': [{'text': message.content}]}
    if isinstance(message, AssistantMessage):
        return {'role': 'assistant', 'content': [{'text': message.content}]}
    if isinstance(message, SystemMessage):
        return {'role': 'system', 'content': [{'text': message.content}]}
    if isinstance(message, UserMultiPartMessage):
        content = []
        for part in message.content:
            if isinstance(part, TextPart):
                content.append({'text': part.text})
            elif isinstance(part, ImageUrlPart):
                content.append({'image': part.image_url.url})
            else:
                raise TypeError(f'Unknown part type: {part}')
        return {'role': 'user', 'content': content}
    allowed_message_type = (UserMessage, AssistantMessage, SystemMessage, UserMultiPartMessage)
    raise MessageTypeError(message, allowed_message_type=allowed_message_type)


class DashScopeMultiModalChat(RemoteChatCompletionModel):
    model_type: ClassVar[str] = 'dashscope_multimodal'

    parameters: DashScopeMultiModalChatParameters
    settings: DashScopeSettings

    def __init__(
        self,
        model: str = 'qwen-vl-max',
        parameters: DashScopeMultiModalChatParameters | None = None,
        settings: DashScopeSettings | None = None,
        http_client: HttpClient | None = None,
    ) -> None:
        parameters = parameters or DashScopeMultiModalChatParameters()
        settings = settings or DashScopeSettings()  # type: ignore
        http_client = http_client or HttpClient()
        super().__init__(parameters=parameters, settings=settings, http_client=http_client)

        self.model = model

    def _get_request_parameters(self, messages: Messages, parameters: DashScopeMultiModalChatParameters) -> HttpxPostKwargs:
        zhipu_messages = [_convert_message_to_multimodal_chat_message(message) for message in messages]
        headers = {
            'Authorization': self.settings.api_key.get_secret_value(),
            'Content-Type': 'application/json',
        }
        params = {
            'input': {
                'messages': zhipu_messages,
            },
            'model': self.model,
            'parameters': parameters.custom_model_dump(),
        }
        return {
            'url': f'{self.settings.api_base}/multimodal-generation/generation',
            'headers': headers,
            'json': params,
        }

    def _get_stream_request_parameters(
        self, messages: Messages, parameters: DashScopeMultiModalChatParameters
    ) -> HttpxPostKwargs:
        http_parameters = self._get_request_parameters(messages, parameters)
        http_parameters['headers']['Accept'] = 'text/event-stream'
        return http_parameters

    def _parse_reponse(self, response: ResponseValue) -> ChatCompletionOutput:
        choice = response['output']['choices'][0]
        return ChatCompletionOutput(
            model_info=self.model_info,
            message=AssistantMessage(content=choice['message']['content'][0]['text']),
            cost=None,
            extra={'usage': response['usage'], 'request_id': response['request_id']},
        )

    @override
    def generate(self, prompt: Prompt, **kwargs: Unpack[DashScopeMultiModalChatParametersDict]) -> ChatCompletionOutput:
        messages = ensure_messages(prompt)
        parameters = self.parameters.clone_with_changes(**kwargs)
        request_parameters = self._get_request_parameters(messages, parameters)
        response = self.http_client.post(request_parameters=request_parameters)
        return self._parse_reponse(response.json())

    @override
    async def async_generate(
        self, prompt: Prompt, **kwargs: Unpack[DashScopeMultiModalChatParametersDict]
    ) -> ChatCompletionOutput:
        messages = ensure_messages(prompt)
        parameters = self.parameters.clone_with_changes(**kwargs)
        request_parameters = self._get_request_parameters(messages, parameters)
        response = await self.http_client.async_post(request_parameters=request_parameters)
        return self._parse_reponse(response.json())

    @override
    def stream_generate(
        self, prompt: Prompt, **kwargs: Unpack[DashScopeMultiModalChatParametersDict]
    ) -> Iterator[ChatCompletionStreamOutput]:
        messages = ensure_messages(prompt)
        parameters = self.parameters.clone_with_changes(**kwargs)
        request_parameters = self._get_stream_request_parameters(messages, parameters)
        message = AssistantMessage(content='')
        is_start = True
        is_finish = False
        for line in self.http_client.stream_post(request_parameters=request_parameters):
            if is_finish:
                continue

            output = self._parse_stream_line(line, message, is_start)
            is_start = False
            is_finish = output.is_finish
            yield output

    @override
    async def async_stream_generate(
        self, prompt: Prompt, **kwargs: Unpack[DashScopeMultiModalChatParametersDict]
    ) -> AsyncIterator[ChatCompletionStreamOutput]:
        messages = ensure_messages(prompt)
        parameters = self.parameters.clone_with_changes(**kwargs)
        request_parameters = self._get_stream_request_parameters(messages, parameters)
        message = AssistantMessage(content='')
        is_start = True
        is_finish = False
        async for line in self.http_client.async_stream_post(request_parameters=request_parameters):
            if is_finish:
                continue

            output = self._parse_stream_line(line, message, is_start)
            is_start = False
            is_finish = output.is_finish
            yield output

    def _parse_stream_line(self, line: str, message: AssistantMessage, is_start: bool) -> ChatCompletionStreamOutput:
        parsed_line = json.loads(line)
        choice = parsed_line['output']['choices'][0]
        finish_reason = choice['finish_reason']
        reply = choice['message']['content'][0]['text']
        usage = parsed_line['usage']
        request_id = parsed_line['request_id']
        extra = {
            'usage': usage,
            'response_id': request_id,
        }
        if finish_reason == 'stop':
            return ChatCompletionStreamOutput(
                model_info=self.model_info,
                cost=None,
                message=AssistantMessage(content=reply),
                extra=extra,
                finish_reason='stop',
                stream=Stream(delta='', control='finish'),
            )
        delta = reply[len(message.content) :]
        message.content = reply
        return ChatCompletionStreamOutput(
            model_info=self.model_info,
            message=message,
            extra=extra,
            stream=Stream(delta=delta, control='start' if is_start else 'continue'),
        )

    @property
    def name(self) -> str:
        return self.model

    @classmethod
    def from_name(cls, name: str) -> Self:
        return cls(model=name)
