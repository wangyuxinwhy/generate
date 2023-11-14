from __future__ import annotations

import os
from typing import Any, ClassVar, Dict, List, Literal, Optional, Union, cast

from pydantic import Field, PositiveInt
from typing_extensions import Annotated, NotRequired, Self, TypedDict, Unpack, override

from generate.chat_completion.function_call import FunctionJsonSchema
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
    ImageUrlPart,
    Message,
    Messages,
    MessageTypeError,
    MessageValueError,
    TextPart,
    ToolCall,
    ToolCallsMessage,
    ToolMessage,
    UserMessage,
    UserMultiPartMessage,
)
from generate.chat_completion.model_output import ChatCompletionModelOutput, FinishStream, Stream
from generate.parameters import ModelParameters
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
    content: Union[str, None, List[Dict]]
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


def convert_to_openai_message(message: Message) -> OpenAIMessage:
    if isinstance(message, UserMessage):
        return {
            'role': 'user',
            'content': message.content,
        }

    if isinstance(message, UserMultiPartMessage):
        content = []
        for part in message.content:
            if isinstance(part, TextPart):
                content.append({'type': 'text', 'text': part.text})
            elif isinstance(part, ImageUrlPart):
                image_url_part_dict = {
                    'type': 'image_url',
                    'image_url': {
                        'url': part.image_url.url,
                    },
                }
                if part.image_url.detail:
                    image_url_part_dict['image_url']['detail'] = part.image_url.detail
                content.append(image_url_part_dict)
            else:
                raise MessageValueError(message, f'OpenAI does not support {type(part)} ')

        return {
            'role': 'user',
            'content': content,
        }

    if isinstance(message, AssistantMessage):
        return {
            'role': 'assistant',
            'content': message.content,
        }

    if isinstance(message, ToolCallsMessage):
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

    if isinstance(message, ToolMessage):
        return {
            'role': 'tool',
            'tool_call_id': message.tool_call_id,
            'content': message.content,
        }

    if isinstance(message, FunctionCallMessage):
        return {
            'role': 'assistant',
            'function_call': {
                'name': message.content.name,
                'arguments': message.content.arguments,
            },
            'content': None,
        }

    if isinstance(message, FunctionMessage):
        return {
            'role': 'function',
            'name': message.name,
            'content': message.content,
        }

    raise MessageTypeError(
        message,
        allowed_message_type=(
            UserMessage,
            AssistantMessage,
            FunctionMessage,
            FunctionCallMessage,
            ToolCallsMessage,
            ToolMessage,
        ),
    )


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


def parse_openai_model_reponse(response: HttpResponse) -> ChatCompletionModelOutput:
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
        extra = {}
        if system_fingerprint := response.get('system_fingerprint'):
            extra['system_fingerprint'] = system_fingerprint

        choice = response['choices'][0]
        if (finish_reason := choice.get('finish_reason')) is None:
            finish_reason = finish_details['type'] if (finish_details := choice.get('finish_details')) else None

        return ChatCompletionModelOutput(
            chat_model_id='openai/' + response['model'],
            messages=messages,
            finish_reason=finish_reason or '',
            usage=response['usage'],
            cost=calculate_cost(response['model'], response['usage']['prompt_tokens'], response['usage']['completion_tokens']),
            extra=extra,
        )


class OpenAIChat(HttpChatModel[OpenAIChatParameters]):
    model_type: ClassVar[str] = 'openai'
    default_api_base: ClassVar[str] = 'https://api.openai.com/v1'

    def __init__(
        self,
        model: str = 'gpt-3.5-turbo',
        api_key: str | None = None,
        api_base: str | None = None,
        system_prompt: str | None = None,
        parameters: OpenAIChatParameters | None = None,
        **kwargs: Unpack[HttpModelInitKwargs],
    ) -> None:
        parameters = parameters or OpenAIChatParameters()
        super().__init__(parameters=parameters, **kwargs)
        self.model = model
        self.system_prompt = system_prompt
        self.api_base = api_base or os.getenv('OPENAI_API_BASE') or self.default_api_base
        self.api_key = api_key or os.environ['OPENAI_API_KEY']

    @override
    def _get_request_parameters(self, messages: Messages, parameters: OpenAIChatParameters) -> HttpxPostKwargs:
        openai_messages = [convert_to_openai_message(message) for message in messages]
        if self.system_prompt:
            openai_messages.insert(0, {'role': 'system', 'content': self.system_prompt})

        headers = {
            'Authorization': f'Bearer {self.api_key}',
        }
        params = {
            'model': self.model,
            'messages': openai_messages,
            **parameters.custom_model_dump(),
        }
        return {
            'url': f'{self.api_base}/chat/completions',
            'headers': headers,
            'json': params,
        }

    @override
    def _parse_reponse(self, response: HttpResponse) -> ChatCompletionModelOutput:
        return parse_openai_model_reponse(response)

    @override
    def _get_stream_request_parameters(self, messages: Messages, parameters: OpenAIChatParameters) -> HttpxPostKwargs:
        http_parameters = self._get_request_parameters(messages, parameters)
        http_parameters['json']['stream'] = True
        return http_parameters

    @override
    def _parse_stream_response(self, response: HttpResponse) -> Stream:
        delta = response['choices'][0]['delta']
        if 'role' in delta:
            return Stream(delta='', control='start')
        if 'content' in delta:
            return Stream(delta=delta['content'], control='continue')

        return FinishStream(finish_reason=response['choices'][0]['finish_reason'])

    @property
    @override
    def name(self) -> str:
        return self.model

    @classmethod
    @override
    def from_name(cls, name: str, **kwargs: Any) -> Self:
        return cls(model=name, **kwargs)
