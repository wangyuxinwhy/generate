from __future__ import annotations

import json
from functools import partial
from typing import Any, AsyncIterator, Callable, ClassVar, Dict, Iterator, List, Literal, Optional, Type, Union, cast

from pydantic import Field, PositiveInt
from typing_extensions import Annotated, NotRequired, Self, TypedDict, Unpack, override

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
    Prompt,
    SystemMessage,
    TextPart,
    ToolCall,
    ToolCallsMessage,
    ToolMessage,
    UserMessage,
    UserMultiPartMessage,
    ensure_messages,
)
from generate.chat_completion.model_output import ChatCompletionOutput, ChatCompletionStreamOutput, Stream
from generate.http import (
    HttpClient,
    HttpxPostKwargs,
    ResponseValue,
)
from generate.model import ModelInfo, ModelParameters, ModelParametersDict
from generate.platforms.openai import OpenAISettings
from generate.types import Probability, Temperature

OpenAIAssistantMessage = Union[AssistantMessage, FunctionCallMessage, ToolCallsMessage]


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


class OpenAIChatParametersDict(ModelParametersDict, total=False):
    temperature: Optional[Temperature]
    top_p: Optional[Probability]
    max_tokens: Optional[PositiveInt]
    functions: Optional[List[FunctionJsonSchema]]
    function_call: Union[Literal['auto'], FunctionCallName, None]
    stop: Union[str, List[str], None]
    presence_penalty: Optional[float]
    frequency_penalty: Optional[float]
    logit_bias: Optional[Dict[int, int]]
    user: Optional[str]
    response_format: Optional[OpenAIResponseFormat]
    seed: Optional[int]
    tools: Optional[List[OpenAITool]]
    tool_choice: Union[Literal['auto'], OpenAIToolChoice, None]


def _to_text_message_dict(role: str, message: Message) -> OpenAIMessage:
    if not isinstance(message.content, str):
        raise TypeError(f'Unexpected message content: {type(message.content)}')
    return {
        'role': role,
        'content': message.content,
    }


def _to_user_multipart_message_dict(message: UserMultiPartMessage) -> OpenAIMessage:
    content = []
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


def convert_openai_message_to_generate_message(
    message: dict[str, Any]
) -> FunctionCallMessage | ToolCallsMessage | AssistantMessage:
    if function_call := message.get('function_call'):
        function_call = cast(OpenAIFunctionCall, function_call)
        return FunctionCallMessage(
            content=FunctionCall(
                name=function_call.get('name') or '',
                arguments=function_call['arguments'],
            ),
        )
    if tool_calls := message.get('tool_calls'):
        tool_calls = cast(List[OpenAIToolCall], tool_calls)
        return ToolCallsMessage(
            content=[
                ToolCall(
                    id=tool_call['id'],
                    function=FunctionCall(
                        name=tool_call['function'].get('name') or '',
                        arguments=tool_call['function']['arguments'],
                    ),
                )
                for tool_call in tool_calls
            ],
        )
    return AssistantMessage(content=message['content'] or '')


def parse_openai_model_reponse(response: ResponseValue) -> ChatCompletionOutput[OpenAIAssistantMessage]:
    message = convert_openai_message_to_generate_message(response['choices'][0]['message'])
    extra = {'usage': response['usage']}
    if system_fingerprint := response.get('system_fingerprint'):
        extra['system_fingerprint'] = system_fingerprint

    choice = response['choices'][0]
    if (finish_reason := choice.get('finish_reason')) is None:
        finish_reason = finish_details['type'] if (finish_details := choice.get('finish_details')) else None

    return ChatCompletionOutput[OpenAIAssistantMessage](
        model_info=ModelInfo(task='chat_completion', type='openai', name=response['model']),
        message=message,
        finish_reason=finish_reason or '',
        cost=calculate_cost(response['model'], response['usage']['prompt_tokens'], response['usage']['completion_tokens']),
        extra=extra,
    )


class _StreamResponseProcessor:
    def __init__(self) -> None:
        self.message: OpenAIAssistantMessage | None = None
        self.is_start = True

    def process(self, response: ResponseValue) -> ChatCompletionStreamOutput[OpenAIAssistantMessage] | None:
        delta_dict = response['choices'][0]['delta']

        if self.message is None:
            self.message = self.process_initial_message(delta_dict)
            if self.message is None:
                return None
        else:
            self.update_existing_message(delta_dict)
        extra = self.extract_extra_info(response)
        cost = cost = self.calculate_response_cost(response)
        finish_reason = self.determine_finish_reason(response)
        stream_control = 'finish' if finish_reason else 'start' if self.is_start else 'continue'
        self.is_start = False
        return ChatCompletionStreamOutput[OpenAIAssistantMessage](
            model_info=ModelInfo(task='chat_completion', type='openai', name=response['model']),
            message=self.message,
            finish_reason=finish_reason,
            cost=cost,
            extra=extra,
            stream=Stream(delta=delta_dict.get('content') or '', control=stream_control),
        )

    def process_initial_message(self, delta_dict: dict[str, Any]) -> OpenAIAssistantMessage | None:
        if (
            delta_dict.get('content') is None
            and delta_dict.get('tool_calls') is None
            and delta_dict.get('function_call') is None
        ):
            return None
        return convert_openai_message_to_generate_message(delta_dict)

    def update_existing_message(self, delta_dict: dict[str, Any]) -> None:
        if not delta_dict:
            return

        if isinstance(self.message, AssistantMessage):
            delta = delta_dict['content']
            self.message.content += delta
        elif isinstance(self.message, FunctionCallMessage):
            self.message.content.arguments += delta_dict['function_call']['arguments']
        elif isinstance(self.message, ToolCallsMessage):
            index = delta_dict['tool_calls'][0]['index']
            if index >= len(self.message.content):
                new_tool_calls_message = cast(ToolCallsMessage, convert_openai_message_to_generate_message(delta_dict))
                self.message.content.append(new_tool_calls_message.content[0])
            else:
                self.message.content[index].function.arguments += delta_dict['tool_calls'][0]['function']['arguments']

    def extract_extra_info(self, response: ResponseValue) -> dict[str, Any]:
        extra = {}
        if usage := response.get('usage'):
            extra['usage'] = usage
        if system_fingerprint := response.get('system_fingerprint'):
            extra['system_fingerprint'] = system_fingerprint
        return extra

    @staticmethod
    def calculate_response_cost(response: ResponseValue) -> float | None:
        if usage := response.get('usage'):
            return calculate_cost(response['model'], usage['prompt_tokens'], usage['completion_tokens'])
        return None

    def determine_finish_reason(self, response: ResponseValue) -> str | None:
        choice = response['choices'][0]
        if (finish_reason := choice.get('finish_reason')) is None:
            finish_reason: str | None = finish_details['type'] if (finish_details := choice.get('finish_details')) else None
        return finish_reason


class OpenAIChat(ChatCompletionModel):
    model_type: ClassVar[str] = 'openai'

    def __init__(
        self,
        model: str = 'gpt-3.5-turbo',
        parameters: OpenAIChatParameters | None = None,
        settings: OpenAISettings | None = None,
        http_client: HttpClient | None = None,
    ) -> None:
        self.model = model
        self.parameters = parameters or OpenAIChatParameters()
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

    @override
    def generate(
        self, prompt: Prompt, **kwargs: Unpack[OpenAIChatParametersDict]
    ) -> ChatCompletionOutput[OpenAIAssistantMessage]:
        messages = ensure_messages(prompt)
        parameters = self.parameters.update_with_validate(**kwargs)
        request_parameters = self._get_request_parameters(messages, parameters)
        response = self.http_client.post(request_parameters)
        return parse_openai_model_reponse(response.json())

    @override
    async def async_generate(
        self, prompt: Prompt, **kwargs: Unpack[OpenAIChatParametersDict]
    ) -> ChatCompletionOutput[OpenAIAssistantMessage]:
        messages = ensure_messages(prompt)
        parameters = self.parameters.update_with_validate(**kwargs)
        request_parameters = self._get_request_parameters(messages, parameters)
        response = await self.http_client.async_post(request_parameters=request_parameters)
        return parse_openai_model_reponse(response.json())

    def _get_stream_request_parameters(self, messages: Messages, parameters: OpenAIChatParameters) -> HttpxPostKwargs:
        http_parameters = self._get_request_parameters(messages, parameters)
        http_parameters['json']['stream'] = True
        return http_parameters

    @override
    def stream_generate(
        self, prompt: Prompt, **kwargs: Unpack[OpenAIChatParametersDict]
    ) -> Iterator[ChatCompletionStreamOutput[OpenAIAssistantMessage]]:
        messages = ensure_messages(prompt)
        parameters = self.parameters.update_with_validate(**kwargs)
        request_parameters = self._get_stream_request_parameters(messages, parameters)
        stream_processor = _StreamResponseProcessor()
        is_finish = False
        for line in self.http_client.stream_post(request_parameters=request_parameters):
            if is_finish:
                continue

            output = stream_processor.process(json.loads(line))
            if output is None:
                continue
            is_finish = output.is_finish
            yield output

    @override
    async def async_stream_generate(
        self, prompt: Prompt, **kwargs: Unpack[OpenAIChatParametersDict]
    ) -> AsyncIterator[ChatCompletionStreamOutput[OpenAIAssistantMessage]]:
        messages = ensure_messages(prompt)
        parameters = self.parameters.update_with_validate(**kwargs)
        request_parameters = self._get_stream_request_parameters(messages, parameters)
        stream_processor = _StreamResponseProcessor()
        is_finish = False
        async for line in self.http_client.async_stream_post(request_parameters=request_parameters):
            if is_finish:
                continue

            output = stream_processor.process(json.loads(line))
            if output is None:
                continue
            is_finish = output.is_finish
            yield output

    @property
    @override
    def name(self) -> str:
        return self.model

    @classmethod
    @override
    def from_name(cls, name: str) -> Self:
        return cls(model=name)
