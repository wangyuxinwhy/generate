from __future__ import annotations

import base64
import json
import uuid
from abc import ABC
from functools import partial
from typing import Any, Callable, Dict, List, Literal, Type, Union, cast

from typing_extensions import NotRequired, TypedDict, override

from generate.chat_completion.base import RemoteChatCompletionModel
from generate.chat_completion.cost_caculator import GeneralCostCalculator
from generate.chat_completion.message import (
    AssistantMessage,
    FunctionCall,
    FunctionMessage,
    ImagePart,
    Message,
    MessageTypeError,
    Prompt,
    SystemMessage,
    TextPart,
    ToolCall,
    ToolMessage,
    UserMessage,
    UserMultiPartMessage,
    ensure_messages,
)
from generate.chat_completion.message.core import Messages
from generate.chat_completion.model_output import ChatCompletionOutput, ChatCompletionStreamOutput
from generate.chat_completion.stream_manager import StreamManager
from generate.chat_completion.tool import FunctionJsonSchema, Tool
from generate.http import (
    HttpxPostKwargs,
    ResponseValue,
)
from generate.model import ModelInfo
from generate.platforms.openai_like import OpenAILikeSettings


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
            if isinstance(part, ImagePart):
                image_format = part.image_format or 'png'
                url: str = f'data:image/{image_format};base64,{base64.b64encode(part.image).decode()}'
                image_url_dict = {'url': url}
            else:
                image_url_dict = {}
                image_url_dict['url'] = part.image_url.url
                if part.image_url.detail:
                    image_url_dict['detail'] = part.image_url.detail
            image_url_part_dict: dict[str, Any] = {
                'type': 'image_url',
                'image_url': image_url_dict,
            }
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


def _to_asssistant_message_dict(message: AssistantMessage) -> OpenAIMessage:
    base_dict = {
        'role': 'assistant',
        'content': message.content or None,
    }
    if message.tool_calls:
        tool_calls = [
            {
                'id': tool_call.id,
                'type': 'function',
                'function': {
                    'name': tool_call.function.name,
                    'arguments': tool_call.function.arguments,
                },
            }
            for tool_call in message.tool_calls
        ]
        base_dict['tool_calls'] = tool_calls
    if message.function_call:
        base_dict['function_call'] = {
            'name': message.function_call.name,
            'arguments': message.function_call.arguments,
        }
    return cast(OpenAIMessage, base_dict)


def _to_function_message_dict(message: FunctionMessage) -> OpenAIMessage:
    return {
        'role': 'function',
        'name': message.name,
        'content': message.content,
    }


def convert_to_openai_message(message: Message) -> OpenAIMessage:
    to_function_map: dict[Type[Message], Callable[[Any], OpenAIMessage]] = {
        SystemMessage: partial(_to_text_message_dict, 'system'),
        UserMessage: partial(_to_text_message_dict, 'user'),
        AssistantMessage: partial(_to_asssistant_message_dict),
        UserMultiPartMessage: _to_user_multipart_message_dict,
        ToolMessage: _to_tool_message_dict,
        FunctionMessage: _to_function_message_dict,
    }
    if to_function := to_function_map.get(type(message)):
        return to_function(message)

    raise MessageTypeError(message, allowed_message_type=tuple(to_function_map.keys()))


def openai_calculate_cost(model_name: str, input_tokens: int, output_tokens: int) -> float | None:
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
    if 'moonshot' in model_name:
        if '8k' in model_name:
            return 0.012 * (input_tokens / 1000) + 0.012 * (output_tokens / 1000)
        if '32k' in model_name:
            return 0.024 * (input_tokens / 1000) + 0.024 * (output_tokens / 1000)
        if '128k' in model_name:
            return 0.06 * (input_tokens / 1000) + 0.06 * (output_tokens / 1000)
    return None


def _convert_to_assistant_message(message: dict[str, Any]) -> AssistantMessage:
    if function_call_dict := message.get('function_call'):
        function_call = FunctionCall(
            name=function_call_dict.get('name') or '',
            arguments=function_call_dict['arguments'],
        )
    else:
        function_call = None

    if tool_calls_dict := message.get('tool_calls'):
        tool_calls = [
            ToolCall(
                id=tool_call['id'],
                function=FunctionCall(
                    name=tool_call['function'].get('name') or '',
                    arguments=tool_call['function']['arguments'],
                ),
            )
            for tool_call in tool_calls_dict
        ]
    else:
        tool_calls = None
    return AssistantMessage(content=message.get('content') or '', function_call=function_call, tool_calls=tool_calls)


def convert_to_openai_tool(tool: Tool) -> OpenAITool:
    return OpenAITool(type='function', function=tool.json_schema)


def process_openai_like_model_reponse(response: ResponseValue, model_type: str) -> ChatCompletionOutput:
    message = _convert_to_assistant_message(response['choices'][0]['message'])
    extra = {'usage': response['usage']}
    if system_fingerprint := response.get('system_fingerprint'):
        extra['system_fingerprint'] = system_fingerprint

    choice = response['choices'][0]
    if (finish_reason := choice.get('finish_reason')) is None:
        finish_reason = finish_details['type'] if (finish_details := choice.get('finish_details')) else None

    try:
        if model_type == 'openai':
            cost = openai_calculate_cost(
                model_name=response['model'],
                input_tokens=response['usage']['prompt_tokens'],
                output_tokens=response['usage']['completion_tokens'],
            )
        else:
            cost_calculator = GeneralCostCalculator()
            cost = cost_calculator.calculate(
                model_type=model_type,
                model_name=response['model'],
                input_tokens=response['usage']['prompt_tokens'],
                output_tokens=response['usage']['completion_tokens'],
            )
    except Exception:
        cost = None

    return ChatCompletionOutput(
        model_info=ModelInfo(task='chat_completion', type=model_type, name=response['model']),
        message=message,
        finish_reason=finish_reason,
        cost=cost,
        extra=extra,
    )


class OpenAILikeChat(RemoteChatCompletionModel, ABC):
    settings: OpenAILikeSettings

    @override
    def _get_request_parameters(self, prompt: Prompt, stream: bool = False, **kwargs: Any) -> HttpxPostKwargs:
        messages = ensure_messages(prompt)
        parameters = self.parameters.clone_with_changes(**kwargs)
        openai_messages = self._convert_to_openai_messages(messages)
        headers = {
            'Authorization': f'Bearer {self.settings.api_key.get_secret_value()}',
        }
        params = {
            'model': self.model,
            'messages': openai_messages,
            **parameters.custom_model_dump(),
        }
        if stream:
            params['stream'] = True

        return {
            'url': f'{self.settings.api_base}/chat/completions',
            'headers': headers,
            'json': params,
        }

    def _convert_to_openai_messages(self, messages: Messages) -> List[OpenAIMessage]:
        return [convert_to_openai_message(message) for message in messages]

    @staticmethod
    def generate_tool_call_id() -> str:
        return f'call_{uuid.uuid4()}'

    @override
    def _process_reponse(self, response: Dict[str, Any]) -> ChatCompletionOutput:
        return process_openai_like_model_reponse(response, model_type=self.model_type)

    @override
    def _process_stream_line(self, line: str, stream_manager: StreamManager) -> ChatCompletionStreamOutput | None:
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            return None

        delta_dict = data['choices'][0].get('delta', {})
        self._update_delta(delta_dict, stream_manager=stream_manager)
        stream_manager.extra = self._extract_extra_info(data)
        stream_manager.cost = self._calculate_cost(data)
        stream_manager.finish_reason = self._determine_finish_reason(data)
        return stream_manager.build_stream_output()

    def _update_delta(self, delta_dict: dict[str, Any], stream_manager: StreamManager) -> None:
        delta_content: str = delta_dict.get('content') or ''
        stream_manager.delta = delta_content

        if delta_dict.get('tool_calls'):
            index = delta_dict['tool_calls'][0]['index']
            if index >= len(stream_manager.tool_calls or []):
                new_tool_calls_message = _convert_to_assistant_message(delta_dict).tool_calls
                assert new_tool_calls_message is not None
                if stream_manager.tool_calls is None:
                    stream_manager.tool_calls = []
                stream_manager.tool_calls.append(new_tool_calls_message[0])
            else:
                assert stream_manager.tool_calls is not None
                stream_manager.tool_calls[index].function.arguments += delta_dict['tool_calls'][0]['function']['arguments']

        if delta_dict.get('function_call'):
            if stream_manager.function_call is None:
                stream_manager.function_call = FunctionCall(name='', arguments='')
            function_name = delta_dict['function_call'].get('name', '')
            stream_manager.function_call.name += function_name
            arguments = delta_dict['function_call'].get('arguments', '')
            stream_manager.function_call.arguments += arguments

    def _extract_extra_info(self, response: ResponseValue) -> dict[str, Any]:
        extra = {
            'id': response['id'],
        }
        choice = response['choices'][0]
        if usage := response.get('usage'):
            extra['usage'] = usage
        if usage := choice.get('usage'):
            extra['usage'] = usage
        if system_fingerprint := response.get('system_fingerprint'):
            extra['system_fingerprint'] = system_fingerprint
        return extra

    def _calculate_cost(self, response: ResponseValue) -> float | None:
        if response.get('usage') is None:
            return None

        if self.model_type == 'openai':
            return openai_calculate_cost(
                model_name=response['model'],
                input_tokens=response['usage']['prompt_tokens'],
                output_tokens=response['usage']['completion_tokens'],
            )

        cost_calculator = GeneralCostCalculator()
        input_tokens = response['usage'].get('prompt_tokens', 0)
        output_tokens = response['usage'].get('completion_tokens', 0)
        if 'total_tokens' in response['usage']:
            input_tokens = 0
            output_tokens = response['usage']['total_tokens']
        return cost_calculator.calculate(
            model_type=self.model_type,
            model_name=response['model'],
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    def _determine_finish_reason(self, response: ResponseValue) -> str | None:
        choice = response['choices'][0]
        finish_reason = choice.get('finish_reason') or None
        if finish_reason is None:
            finish_reason: str | None = finish_details['type'] if (finish_details := choice.get('finish_details')) else None
        return finish_reason
