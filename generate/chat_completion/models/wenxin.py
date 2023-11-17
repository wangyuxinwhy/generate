from __future__ import annotations

import json
from typing import Any, AsyncIterator, ClassVar, Iterator, List, Literal, Optional

from pydantic import Field, model_validator
from typing_extensions import Annotated, NotRequired, Self, TypedDict, override

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
    HttpxPostKwargs,
    UnexpectedResponseError,
)
from generate.model import ModelParameters
from generate.platforms.baidu import QianfanSettings, QianfanTokenMixin
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
    user: Optional[str] = Field(default=None, serialization_alias='user_id')

    @model_validator(mode='after')
    def system_function_conflict(self) -> Self:
        if self.system is not None and self.functions is not None:
            raise ValueError('system and functions cannot be used together')
        return self

    def custom_model_dump(self) -> dict[str, Any]:
        output = super().custom_model_dump()
        if 'temperature' in output:
            output['temperature'] = max(0.01, output['temperature'])
        return output


class WenxinChat(ChatCompletionModel[WenxinChatParameters], QianfanTokenMixin):
    model_type: ClassVar[str] = 'wenxin'
    model_name_entrypoint_map: ClassVar[dict[str, str]] = {
        'ERNIE-Bot': 'completions',
        'ERNIE-Bot-turbo': 'eb-instant',
        'ERNIE-Bot-4': 'completions_pro',
    }

    def __init__(
        self,
        model: str = 'ERNIE-Bot',
        settings: QianfanSettings | None = None,
        parameters: WenxinChatParameters | None = None,
        http_client: HttpClient | None = None,
    ) -> None:
        parameters = parameters or WenxinChatParameters()
        super().__init__(parameters=parameters)

        self._token = None
        self.model = model
        self.settings = settings or QianfanSettings()  # type: ignore
        self.http_client = http_client or HttpClient()

    def _get_request_parameters(self, messages: Messages, parameters: WenxinChatParameters) -> HttpxPostKwargs:
        wenxin_messages: list[WenxinMessage] = [convert_to_wenxin_message(message) for message in messages]
        parameters_dict = parameters.custom_model_dump()
        if 'temperature' in parameters_dict:
            parameters_dict['temperature'] = max(0.01, parameters_dict['temperature'])
        json_data = {'messages': wenxin_messages, **parameters_dict}

        return {
            'url': self.settings.comlpetion_api_base + self.model_name_entrypoint_map[self.model],
            'json': json_data,
            'params': {'access_token': self.token},
            'headers': {'Content-Type': 'application/json'},
        }

    def _completion(self, messages: Messages, parameters: WenxinChatParameters) -> ChatCompletionOutput:
        request_parameters = self._get_request_parameters(messages, parameters)
        response = self.http_client.post(request_parameters)
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
        for line in self.http_client.stream_post(request_parameters=request_parameters):
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
        async for line in self.http_client.async_stream_post(request_parameters=request_parameters):
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

    def calculate_cost(self, usage: dict[str, Any]) -> float | None:
        if self.name == 'ERNIE-Bot':
            return 0.012 * (usage['total_tokens'] / 1000)
        if self.name == 'ERNIE-Bot-turbo':
            return 0.008 * (usage['total_tokens'] / 1000)
        if self.name == 'ERNIE-Bot-4':
            return 0.12 * (usage['total_tokens'] / 1000)
        return None

    @property
    @override
    def name(self) -> str:
        return self.model

    @classmethod
    @override
    def from_name(cls, name: str, **kwargs: Any) -> Self:
        return cls(model=name, **kwargs)
