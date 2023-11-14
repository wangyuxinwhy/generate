from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Any, ClassVar, List, Literal, Optional

import httpx
from pydantic import Field, field_validator, model_validator
from typing_extensions import Annotated, NotRequired, Self, TypedDict, Unpack, override

from generate.chat_completion.http_chat import (
    HttpChatModel,
    HttpModelInitKwargs,
    HttpResponse,
    HttpxPostKwargs,
    UnexpectedResponseError,
)
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
from generate.chat_completion.model_output import ChatCompletionModelOutput, FinishStream, Stream
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


class WenxinChat(HttpChatModel[WenxinChatParameters]):
    model_type: ClassVar[str] = 'wenxin'
    model_name_entrypoint_map: ClassVar[dict[str, str]] = {
        'llama_2_7b': 'llama_2_7b',
        'llama_2_13b': 'llama_2_13b',
        'llama_2_70b': 'llama_2_70b',
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
        **kwargs: Unpack[HttpModelInitKwargs],
    ) -> None:
        parameters = parameters or WenxinChatParameters()
        super().__init__(parameters=parameters, **kwargs)
        self.model = self.normalize_model(model)
        self.api_base = api_base or self.default_api_base
        self._api_key = api_key or os.environ['WENXIN_API_KEY']
        self._secret_key = secret_key or os.environ['WENXIN_SECRET_KEY']
        self._access_token = self.get_access_token()
        self._access_token_expires_at = datetime.now() + timedelta(days=self.access_token_refresh_days)

    @property
    @override
    def name(self) -> str:
        return self.model

    @property
    def api_url(self) -> str:
        return self.api_base + self.model_name_entrypoint_map[self.model]

    @staticmethod
    def normalize_model(model: str) -> str:
        _map = {
            'llama-2-7b-chat': 'llama_2_7b',
            'llama-2-13b-chat': 'llama_2_13b',
            'llama-2-70b-chat': 'llama_2_70b',
        }
        return _map.get(model, model)

    def get_access_token(self) -> str:
        headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
        params = {'grant_type': 'client_credentials', 'client_id': self._api_key, 'client_secret': self._secret_key}
        response = httpx.post(self.access_token_url, headers=headers, params=params)
        response.raise_for_status()
        response_dict = response.json()
        if 'error' in response_dict:
            raise UnexpectedResponseError(response_dict)
        return response_dict['access_token']

    @override
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

    @override
    def _get_stream_request_parameters(self, messages: Messages, parameters: WenxinChatParameters) -> HttpxPostKwargs:
        http_parameters = self._get_request_parameters(messages, parameters)
        http_parameters['json']['stream'] = True
        return http_parameters

    @override
    def _parse_stream_response(self, response: HttpResponse) -> Stream:
        if response['is_end']:
            return FinishStream(
                delta=response['result'],
                usage=response['usage'],
                cost=self.calculate_cost(response['usage']),
                extra={
                    'is_truncated': response['is_truncated'],
                    'need_clear_history': response['need_clear_history'],
                },
            )
        return Stream(delta=response['result'], control='continue')

    @override
    def _parse_reponse(self, response: HttpResponse) -> ChatCompletionModelOutput:
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
        return ChatCompletionModelOutput(
            chat_model_id=self.model_id,
            messages=messages,
            usage=response['usage'],
            cost=self.calculate_cost(response['usage']),
            extra={
                'is_truncated': response['is_truncated'],
                'need_clear_history': response['need_clear_history'],
            },
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
