from __future__ import annotations

import json
from functools import partial
from typing import Any, AsyncIterator, Callable, ClassVar, Dict, Iterator, List, Literal, Optional, Type, Union, cast

from pydantic import Field, PositiveInt
from typing_extensions import Annotated, NotRequired, Self, TypedDict, override

from generate.chat_completion.base import ChatCompletionModel
from generate.chat_completion.function_call import FunctionJsonSchema
from generate.chat_completion.message import (
    AssistantMessage,
    FunctionCall,
    FunctionCallMessage,
    FunctionMessage,
    Message,
    Messages,
    MessageTypeError,
    SystemMessage,
    TextPart,
    ToolCall,
    ToolCallsMessage,
    ToolMessage,
    UserMessage,
    UserMultiPartMessage,
)
from generate.chat_completion.model_output import ChatCompletionOutput, ChatCompletionStreamOutput, Stream
from generate.http import (
    HttpClient,
    HttpxPostKwargs,
    UnexpectedResponseError,
)
from generate.model import ModelInfo, ModelParameters
from generate.platforms.openai import OpenAISettings
from generate.types import Probability, Temperature


class FunctionCallName(TypedDict):
    name: str


class OpenAIFunctionCall(TypedDict):
    name: str
    arguments: str


class OpenAITool(TypedDict):
    type: Literal['function']
    function: FunctionJsonSchema


class OpenAIToolChoice(TypedDict):
    type: Literal['function']
    function: FunctionCallName


class OpenAIToolCall(TypedDict):
    id: str
    type: Literal['function']
    function: OpenAIFunctionCall


class OpenAIMessage(TypedDict):
    role: str
    content: Union[str, None, List[Dict[str, Any]]]
    name: NotRequired[str]
    function_call: NotRequired[OpenAIFunctionCall]
    tool_call_id: NotRequired[str]
    tool_calls: NotRequired[List[OpenAIToolCall]]


class OpenAIResponseFormat(TypedDict):
    type: Literal['json_object', 'text']


class OpenAIChatParameters(ModelParameters):
    temperature: Optional[Temperature] = None
    top_p: Optional[Probability] = None
    max_tokens: Optional[PositiveInt] = None
    functions: Optional[List[FunctionJsonSchema]] = None
    function_call: Union[Literal['auto'], FunctionCallName, None] = None
    stop: Union[str, List[str], None] = None
    presence_penalty: Optional[Annotated[float, Field(ge=-2, le=2)]] = None
    frequency_penalty: Optional[Annotated[float, Field(ge=-2, le=2)]] = None
    logit_bias: Optional[Dict[int, Annotated[int, Field(ge=-100, le=100)]]] = None
    user: Optional[str] = None
    response_format: Optional[OpenAIResponseFormat] = None
    seed: Optional[int] = None
    tools: Optional[List[OpenAITool]] = None
    tool_choice: Union[Literal['auto'], OpenAIToolChoice, None] = None


def _to_text_message_dict(role: str, message: Message) -> OpenAIMessage:
    return {
        'role': role,
        'content': message.content,
    }


def _to_user_multipart_message_dict(message: UserMultiPartMessage) -> OpenAIMessage:
    content: list[dict[str, Any]] = []
    for part in message.content:
        if isinstance(part, TextPart):
            content.append({'type': 'text', 'text': part.text})
        else:
            image_url_part_dict: dict[str, Any] = {
                'type': 'image_url',
                'image_url': {
                    'url': part.image_url.url,
                },
            }
            if part.image_url.detail:
                image_url_part_dict['image_url']['detail'] = part.image_url.detail
            content.append(image_url_part_dict)
    return {
        'role': 'user',
        'content': content,
    }


def _to_tool_message_dict(message: ToolMessage) -> OpenAIMessage:
    return {
        'role': 'tool',
        'tool_call_id': message.tool_call_id,
        'content': message.content,
    }


def _to_tool_calls_message_dict(message: ToolCallsMessage) -> OpenAIMessage:
    return {
        'role': 'assistant',
        'content': None,
        'tool_calls': [
            {
                'id': tool_call.id,
                'type': 'function',
                'function': {
                    'name': tool_call.function.name,
                    'arguments': tool_call.function.arguments,
                },
            }
            for tool_call in message.content
        ],
    }


def _to_function_message_dict(message: FunctionMessage) -> OpenAIMessage:
    return {
        'role': 'function',
        'name': message.name,
        'content': message.content,
    }


def _to_function_call_message_dict(message: FunctionCallMessage) -> OpenAIMessage:
    return {
        'role': 'assistant',
        'function_call': {
            'name': message.content.name,
            'arguments': message.content.arguments,
        },
        'content': None,
    }


def convert_to_openai_message(message: Message) -> OpenAIMessage:
    to_function_map: dict[Type[Message], Callable[[Any], OpenAIMessage]] = {
        SystemMessage: partial(_to_text_message_dict, 'system'),
        UserMessage: partial(_to_text_message_dict, 'user'),
        AssistantMessage: partial(_to_text_message_dict, 'assistant'),
        UserMultiPartMessage: _to_user_multipart_message_dict,
        ToolMessage: _to_tool_message_dict,
        ToolCallsMessage: _to_tool_calls_message_dict,
        FunctionMessage: _to_function_message_dict,
        FunctionCallMessage: _to_function_call_message_dict,
    }
    if to_function := to_function_map.get(type(message)):
        return to_function(message)

    raise MessageTypeError(message, allowed_message_type=tuple(to_function_map.keys()))


def calculate_cost(model_name: str, input_tokens: int, output_tokens: int) -> float | None:
    dollar_to_yuan = 7
    if model_name in ('gpt-4-1106-preview', 'gpt-4-1106-vision-preview'):
        return (0.01 * dollar_to_yuan) * (input_tokens / 1000) + (0.03 * dollar_to_yuan) * (output_tokens / 1000)
    if 'gpt-4-turbo' in model_name:
        return (0.01 * dollar_to_yuan) * (input_tokens / 1000) + (0.03 * dollar_to_yuan) * (output_tokens / 1000)
    if 'gpt-4-32k' in model_name:
        return (0.06 * dollar_to_yuan) * (input_tokens / 1000) + (0.12 * dollar_to_yuan) * (output_tokens / 1000)
    if 'gpt-4' in model_name:
        return (0.03 * dollar_to_yuan) * (input_tokens / 1000) + (0.06 * dollar_to_yuan) * (output_tokens / 1000)
    if 'gpt-3.5-turbo' in model_name:
        return (0.001 * dollar_to_yuan) * (input_tokens / 1000) + (0.002 * dollar_to_yuan) * (output_tokens / 1000)
    return None


def parse_openai_model_reponse(response: dict[str, Any]) -> ChatCompletionOutput:
    message = response['choices'][0]['message']
    try:
        if function_call := message.get('function_call'):
            function_call = cast(OpenAIFunctionCall, function_call)
            messages = [
                FunctionCallMessage(
                    content=FunctionCall(
                        name=function_call['name'],
                        arguments=function_call['arguments'],
                    ),
                )
            ]
        elif tool_calls := message.get('tool_calls'):
            tool_calls = cast(List[OpenAIToolCall], tool_calls)
            messages = [
                ToolCallsMessage(
                    content=[
                        ToolCall(
                            id=tool_call['id'],
                            function=FunctionCall(
                                name=tool_call['function']['name'],
                                arguments=tool_call['function']['arguments'],
                            ),
                        )
                        for tool_call in tool_calls
                    ],
                )
            ]
        else:
            messages = [AssistantMessage(content=message['content'])]
    except (KeyError, IndexError) as e:
        raise UnexpectedResponseError(response) from e
    else:
        extra = {'usage': response['usage']}
        if system_fingerprint := response.get('system_fingerprint'):
            extra['system_fingerprint'] = system_fingerprint

        choice = response['choices'][0]
        if (finish_reason := choice.get('finish_reason')) is None:
            finish_reason = finish_details['type'] if (finish_details := choice.get('finish_details')) else None

        return ChatCompletionOutput(
            model_info=ModelInfo(task='chat_completion', type='openai', name=response['model']),
            messages=messages,
            finish_reason=finish_reason or '',
            cost=calculate_cost(response['model'], response['usage']['prompt_tokens'], response['usage']['completion_tokens']),
            extra=extra,
        )


class OpenAIChat(ChatCompletionModel[OpenAIChatParameters]):
    model_type: ClassVar[str] = 'openai'

    def __init__(
        self,
        model: str = 'gpt-3.5-turbo',
        settings: OpenAISettings | None = None,
        parameters: OpenAIChatParameters | None = None,
        http_client: HttpClient | None = None,
    ) -> None:
        parameters = parameters or OpenAIChatParameters()
        super().__init__(parameters=parameters)

        self.model = model
        self.settings = settings or OpenAISettings()  # type: ignore
        self.http_client = http_client or HttpClient()

    def _get_request_parameters(self, messages: Messages, parameters: OpenAIChatParameters) -> HttpxPostKwargs:
        openai_messages = [convert_to_openai_message(message) for message in messages]
        headers = {
            'Authorization': f'Bearer {self.settings.api_key.get_secret_value()}',
        }
        params = {
            'model': self.model,
            'messages': openai_messages,
            **parameters.custom_model_dump(),
        }
        return {
            'url': f'{self.settings.api_base}/chat/completions',
            'headers': headers,
            'json': params,
        }

    def _completion(self, messages: Messages, parameters: OpenAIChatParameters) -> ChatCompletionOutput:
        request_parameters = self._get_request_parameters(messages, parameters)
        response = self.http_client.post(request_parameters)
        return parse_openai_model_reponse(response.json())

    async def _async_completion(self, messages: Messages, parameters: OpenAIChatParameters) -> ChatCompletionOutput:
        request_parameters = self._get_request_parameters(messages, parameters)
        response = await self.http_client.async_post(request_parameters=request_parameters)
        return parse_openai_model_reponse(response.json())

    def _get_stream_request_parameters(self, messages: Messages, parameters: OpenAIChatParameters) -> HttpxPostKwargs:
        http_parameters = self._get_request_parameters(messages, parameters)
        http_parameters['json']['stream'] = True
        return http_parameters

    def _stream_completion(self, messages: Messages, parameters: OpenAIChatParameters) -> Iterator[ChatCompletionStreamOutput]:
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
        self, messages: Messages, parameters: OpenAIChatParameters
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
        delta = parsed_line['choices'][0]['delta']
        if 'content' in delta:
            return ChatCompletionStreamOutput(
                model_info=self.model_info,
                stream=Stream(delta=delta['content'], control='continue'),
            )
        return ChatCompletionStreamOutput(
            model_info=self.model_info,
            stream=Stream(delta='', control='finish'),
            finish_reason=parsed_line['choices'][0]['finish_reason'],
        )

    @property
    @override
    def name(self) -> str:
        return self.model

    @classmethod
    @override
    def from_name(cls, name: str, **kwargs: Any) -> Self:
        return cls(model=name, **kwargs)
