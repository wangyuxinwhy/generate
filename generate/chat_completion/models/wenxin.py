from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from typing import Any, AsyncIterator, ClassVar, Iterator, List, Literal, Optional

import httpx
from pydantic import Field, field_validator, model_validator
from typing_extensions import Annotated, NotRequired, Self, TypedDict, Unpack, override

from generate.chat_completion.base import ChatCompletionModel
from generate.chat_completion.message import (
    AssistantMessage,
    FunctionCall,
    FunctionCallMessage,
    FunctionMessage,
    Message,
    Messages,
    MessageTypeError,
    UserMessage,
)
from generate.chat_completion.model_output import ChatCompletionOutput, ChatCompletionStreamOutput, Stream
from generate.http import (
    HttpClient,
    HttpClientInitKwargs,
    HttpMixin,
    HttpStreamClient,
    HttpxPostKwargs,
    UnexpectedResponseError,
)
from generate.parameters import ModelParameters
from generate.types import JsonSchema, Probability, Temperature


class WenxinMessage(TypedDict):
    role: Literal['user', 'assistant', 'function']
    content: str
    name: NotRequired[str]
    function_call: NotRequired[WenxinFunctionCall]


class WenxinFunctionCall(TypedDict):
    name: str
    arguments: str
    thoughts: NotRequired[str]


class WenxinFunction(TypedDict):
    name: str
    description: str
    parameters: JsonSchema
    responses: NotRequired[JsonSchema]
    examples: NotRequired[List[WenxinMessage]]


def convert_to_wenxin_message(message: Message) -> WenxinMessage:
    if isinstance(message, UserMessage):
        return {
            'role': 'user',
            'content': message.content,
        }

    if isinstance(message, AssistantMessage):
        return {
            'role': 'assistant',
            'content': message.content,
        }

    if isinstance(message, FunctionCallMessage):
        return {
            'role': 'assistant',
            'function_call': {
                'name': message.content.name,
                'arguments': message.content.arguments,
                'thoughts': message.content.thoughts or '',
            },
            'content': '',
        }
    if isinstance(message, FunctionMessage):
        return {
            'role': message.role,
            'name': message.name,
            'content': message.content,
        }

    raise MessageTypeError(message, allowed_message_type=(UserMessage, AssistantMessage, FunctionMessage, FunctionCallMessage))


class WenxinChatParameters(ModelParameters):
    temperature: Optional[Temperature] = None
    top_p: Optional[Probability] = None
    functions: Optional[List[WenxinFunction]] = None
    penalty_score: Optional[Annotated[float, Field(ge=1, le=2)]] = None
    system: Optional[str] = None
    user_id: Optional[str] = None

    @model_validator(mode='after')
    def system_function_conflict(self) -> Self:
        if self.system is not None and self.functions is not None:
            raise ValueError('system and functions cannot be used together')
        return self

    @field_validator('temperature', mode='after')
    @classmethod
    def temperature_gt_0(cls, value: float) -> float | Any:
        if value == 0:
            return 0.01
        return value


class WenxinChat(ChatCompletionModel[WenxinChatParameters], HttpMixin):
    model_type: ClassVar[str] = 'wenxin'
    model_name_entrypoint_map: ClassVar[dict[str, str]] = {
        'ERNIE-Bot': 'completions',
        'ERNIE-Bot-turbo': 'eb-instant',
        'ERNIE-Bot-4': 'completions_pro',
    }
    access_token_refresh_days: ClassVar[int] = 20
    access_token_url: ClassVar[str] = 'https://aip.baidubce.com/oauth/2.0/token'
    default_api_base: ClassVar[str] = 'https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/'

    def __init__(
        self,
        model: str = 'ERNIE-Bot',
        api_key: str | None = None,
        api_base: str | None = None,
        secret_key: str | None = None,
        parameters: WenxinChatParameters | None = None,
        **kwargs: Unpack[HttpClientInitKwargs],
    ) -> None:
        parameters = parameters or WenxinChatParameters()
        super().__init__(parameters=parameters)
        self.model = model
        self.api_base = api_base or self.default_api_base
        self._api_key = api_key or os.environ['WENXIN_API_KEY']
        self._secret_key = secret_key or os.environ['WENXIN_SECRET_KEY']
        self._access_token = self.get_access_token()
        self._access_token_expires_at = datetime.now() + timedelta(days=self.access_token_refresh_days)
        self.http_client = HttpClient(**kwargs)
        self.http_stream_client = HttpStreamClient(**kwargs)

    @property
    @override
    def name(self) -> str:
        return self.model

    @property
    def api_url(self) -> str:
        return self.api_base + self.model_name_entrypoint_map[self.model]

    def get_access_token(self) -> str:
        headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
        params = {'grant_type': 'client_credentials', 'client_id': self._api_key, 'client_secret': self._secret_key}
        response = httpx.post(self.access_token_url, headers=headers, params=params)
        response.raise_for_status()
        response_dict = response.json()
        if 'error' in response_dict:
            raise UnexpectedResponseError(response_dict)
        return response_dict['access_token']

    def _get_request_parameters(self, messages: Messages, parameters: WenxinChatParameters) -> HttpxPostKwargs:
        self.maybe_refresh_access_token()

        wenxin_messages: list[WenxinMessage] = [convert_to_wenxin_message(message) for message in messages]
        parameters_dict = parameters.model_dump(exclude_none=True)
        if 'temperature' in parameters_dict:
            parameters_dict['temperature'] = max(0.01, parameters_dict['temperature'])
        json_data = {'messages': wenxin_messages, **parameters_dict}

        return {
            'url': self.api_url,
            'json': json_data,
            'params': {'access_token': self._access_token},
            'headers': {'Content-Type': 'application/json'},
        }

    def _completion(self, messages: Messages, parameters: WenxinChatParameters) -> ChatCompletionOutput:
        request_parameters = self._get_request_parameters(messages, parameters)
        response = self.http_client.post(request_parameters=request_parameters)
        return self._parse_reponse(response.json())

    async def _async_completion(self, messages: Messages, parameters: WenxinChatParameters) -> ChatCompletionOutput:
        request_parameters = self._get_request_parameters(messages, parameters)
        response = await self.http_client.async_post(request_parameters=request_parameters)
        return self._parse_reponse(response.json())

    def _parse_reponse(self, response: dict[str, Any]) -> ChatCompletionOutput:
        if response.get('error_msg'):
            raise UnexpectedResponseError(response)
        if response.get('function_call'):
            messages = [
                FunctionCallMessage(
                    content=FunctionCall(
                        name=response['function_call']['name'],
                        arguments=response['function_call']['arguments'],
                        thoughts=response['function_call']['thoughts'],
                    ),
                )
            ]
        else:
            messages = [AssistantMessage(content=response['result'])]
        return ChatCompletionOutput(
            model_info=self.model_info,
            messages=messages,
            cost=self.calculate_cost(response['usage']),
            extra={
                'is_truncated': response['is_truncated'],
                'need_clear_history': response['need_clear_history'],
                'usage': response['usage'],
            },
        )

    def _get_stream_request_parameters(self, messages: Messages, parameters: WenxinChatParameters) -> HttpxPostKwargs:
        http_parameters = self._get_request_parameters(messages, parameters)
        http_parameters['json']['stream'] = True
        return http_parameters

    def _stream_completion(self, messages: Messages, parameters: WenxinChatParameters) -> Iterator[ChatCompletionStreamOutput]:
        request_parameters = self._get_stream_request_parameters(messages, parameters)
        yield ChatCompletionStreamOutput(
            model_info=self.model_info,
            stream=Stream(delta='', control='start'),
        )
        reply = ''
        for line in self.http_stream_client.post(request_parameters=request_parameters):
            output = self._parse_stream_line(line)
            reply += output.stream.delta
            if output.is_finish:
                output.messages = [AssistantMessage(content=reply)]
            yield output
            if output.is_finish:
                break

    async def _async_stream_completion(
        self, messages: Messages, parameters: WenxinChatParameters
    ) -> AsyncIterator[ChatCompletionStreamOutput]:
        request_parameters = self._get_stream_request_parameters(messages, parameters)
        yield ChatCompletionStreamOutput(
            model_info=self.model_info,
            stream=Stream(delta='', control='start'),
        )
        reply = ''
        async for line in self.http_stream_client.async_post(request_parameters=request_parameters):
            output = self._parse_stream_line(line)
            reply += output.stream.delta
            if output.is_finish:
                output.messages = [AssistantMessage(content=reply)]
            yield output
            if output.is_finish:
                break

    def _parse_stream_line(self, line: str) -> ChatCompletionStreamOutput:
        parsed_line = json.loads(line)
        if parsed_line['is_end']:
            return ChatCompletionStreamOutput(
                model_info=self.model_info,
                cost=self.calculate_cost(parsed_line['usage']),
                extra={
                    'is_truncated': parsed_line['is_truncated'],
                    'need_clear_history': parsed_line['need_clear_history'],
                    'usage': parsed_line['usage'],
                },
                stream=Stream(delta=parsed_line['result'], control='finish'),
            )
        return ChatCompletionStreamOutput(
            model_info=self.model_info,
            stream=Stream(delta=parsed_line['result'], control='continue'),
        )

    def maybe_refresh_access_token(self) -> None:
        if self._access_token_expires_at < datetime.now():
            self._access_token = self.get_access_token()
            self._access_token_expires_at = datetime.now() + timedelta(days=self.access_token_refresh_days)

    def calculate_cost(self, usage: dict[str, Any]) -> float | None:
        if self.name == 'ERNIE-Bot':
            return 0.012 * (usage['total_tokens'] / 1000)
        if self.name == 'ERNIE-Bot-turbo':
            return 0.008 * (usage['total_tokens'] / 1000)
        if self.name == 'ERNIE-Bot-4':
            return 0.12 * (usage['total_tokens'] / 1000)
        return None

    @classmethod
    @override
    def from_name(cls, name: str, **kwargs: Any) -> Self:
        return cls(model=name, **kwargs)
