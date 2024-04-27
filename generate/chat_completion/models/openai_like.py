from __future__ import annotations

import base64
from abc import ABC
from typing import Any, Dict, List, Literal, Union

from typing_extensions import NotRequired, TypedDict, override

from generate.chat_completion.base import RemoteChatCompletionModel
from generate.chat_completion.cost_caculator import CostCalculator
from generate.chat_completion.message import (
    AssistantMessage,
    FunctionCall,
    FunctionMessage,
    ImagePart,
    SystemMessage,
    TextPart,
    ToolCall,
    ToolMessage,
    UserMessage,
    UserMultiPartMessage,
)
from generate.chat_completion.message.converter import MessageConverter
from generate.chat_completion.message.core import Messages
from generate.chat_completion.model_output import ChatCompletionOutput, ChatCompletionStreamOutput, FinishReason, Usage
from generate.chat_completion.stream_manager import StreamManager
from generate.chat_completion.tool import FunctionJsonSchema, Tool
from generate.http import (
    HttpClient,
    HttpGetKwargs,
    HttpxPostKwargs,
    ResponseValue,
)
from generate.model import ModelInfo, ModelParameters
from generate.platforms.base import PlatformSettings
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


class OpenAIMessageConverter(MessageConverter):
    def convert_system_message(self, message: SystemMessage) -> Dict[str, Any]:
        return {
            'role': 'system',
            'content': message.content,
        }

    def convert_user_message(self, message: UserMessage) -> Dict[str, Any]:
        return {
            'role': 'user',
            'content': message.content,
        }

    def convert_user_multi_part_message(self, message: UserMultiPartMessage) -> Dict[str, Any]:
        content = []
        for part in message.content:
            if isinstance(part, TextPart):
                content.append({'type': 'text', 'text': part.text})
            else:
                if isinstance(part, ImagePart):
                    url: str = f'data:image/{part.image_format};base64,{base64.b64encode(part.image).decode()}'
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

    def convert_tool_message(self, message: ToolMessage) -> Dict[str, Any]:
        return {
            'role': 'tool',
            'tool_call_id': message.tool_call_id,
            'content': message.content,
        }

    def convert_assistant_message(self, message: AssistantMessage) -> Dict[str, Any]:
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
        return base_dict

    def convert_function_message(self, message: FunctionMessage) -> Dict[str, Any]:
        return {
            'role': 'function',
            'name': message.name,
            'content': message.content,
        }


# def openai_calculate_cost(model_name: str, input_tokens: int, output_tokens: int) -> float | None:

#     dollar_to_yuan = 7
#     if model_name in ('gpt-4-1106-preview', 'gpt-4-1106-vision-preview'):
#         return (0.01 * dollar_to_yuan) * (input_tokens / 1000) + (0.03 * dollar_to_yuan) * (output_tokens / 1000)
#     if 'gpt-4-turbo' in model_name:
#         return (0.01 * dollar_to_yuan) * (input_tokens / 1000) + (0.03 * dollar_to_yuan) * (output_tokens / 1000)
#     if 'gpt-4-32k' in model_name:
#         return (0.06 * dollar_to_yuan) * (input_tokens / 1000) + (0.12 * dollar_to_yuan) * (output_tokens / 1000)
#     if 'gpt-4' in model_name:
#         return (0.03 * dollar_to_yuan) * (input_tokens / 1000) + (0.06 * dollar_to_yuan) * (output_tokens / 1000)
#     if 'gpt-3.5-turbo' in model_name:
#         return (0.001 * dollar_to_yuan) * (input_tokens / 1000) + (0.002 * dollar_to_yuan) * (output_tokens / 1000)
#     if 'moonshot' in model_name:
#         if '8k' in model_name:
#             return 0.012 * (input_tokens / 1000) + 0.012 * (output_tokens / 1000)
#         if '32k' in model_name:
#             return 0.024 * (input_tokens / 1000) + 0.024 * (output_tokens / 1000)
#         if '128k' in model_name:
#             return 0.06 * (input_tokens / 1000) + 0.06 * (output_tokens / 1000)
#     return None


def parse_message_dict(message: dict[str, Any]) -> AssistantMessage:
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
    choice = response['choices'][0]
    message = parse_message_dict(choice['message'])
    extra = {'response': response}
    if system_fingerprint := response.get('system_fingerprint'):
        extra['system_fingerprint'] = system_fingerprint

    if (finish_reason := choice.get('finish_reason')) is None:
        finish_reason = finish_details['type'] if (finish_details := choice.get('finish_details')) else None

    if finish_reason:
        finish_reason = FinishReason(finish_reason)
    input_tokens = response['usage']['prompt_tokens']
    output_tokens = response['usage']['completion_tokens']
    cost = None
    for k, v in response['usage'].items():
        if k in ('cost', 'total_cost'):
            cost = v
            break

    return ChatCompletionOutput(
        model_info=ModelInfo(task='chat_completion', type=model_type, name=response['model']),
        message=message,
        finish_reason=finish_reason,
        usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens, cost=cost),
        extra=extra,
    )


class OpenAILikeChat(RemoteChatCompletionModel, ABC):
    settings: OpenAILikeSettings

    def __init__(
        self,
        model: str,
        parameters: ModelParameters,
        settings: PlatformSettings,
        http_client: HttpClient | None = None,
        message_converter: MessageConverter | None = None,
        cost_calculator: CostCalculator | None = None,
    ) -> None:
        http_client = http_client or HttpClient()
        message_converter = message_converter or OpenAIMessageConverter()
        super().__init__(
            model=model,
            parameters=parameters,
            settings=settings,
            http_client=http_client,
            message_converter=message_converter,
            cost_calculator=cost_calculator,
        )

    @override
    def _get_request_parameters(self, messages: Messages, stream: bool = False, **kwargs: Any) -> HttpxPostKwargs:
        parameters = self.parameters.clone_with_changes(**kwargs)
        headers = {
            'Authorization': f'Bearer {self.settings.api_key.get_secret_value()}',
        }
        params = {
            'model': self.model,
            'messages': self.message_converter.convert_messages(messages),
            **parameters.custom_model_dump(),
        }
        if stream:
            params['stream'] = True

        return {
            'url': f'{self.settings.api_base}/chat/completions',
            'headers': headers,
            'json': params,
        }

    @override
    def _process_reponse(self, response: dict[str, Any]) -> ChatCompletionOutput:
        return ChatCompletionOutput(
            model_info=ModelInfo(task='chat_completion', type=self.model_type, name=response['model']),
            message=self._parse_assistant_message(response),
            finish_reason=self._parse_finish_reason(response),
            usage=self._parse_usage(response),
            extra=self._parse_extra_info(response),
        )

    @override
    def _process_stream_response(
        self, response: Dict[str, Any], stream_manager: StreamManager
    ) -> ChatCompletionStreamOutput | None:
        delta_dict = response['choices'][0].get('delta', {})
        self._update_delta(delta_dict, stream_manager=stream_manager)
        stream_manager.extra = self._parse_extra_info(response)
        stream_manager.usage = self._parse_usage(response)
        stream_manager.finish_reason = self._parse_finish_reason(response)
        return stream_manager.build_stream_output()

    def _parse_assistant_message(self, response: dict[str, Any]) -> AssistantMessage:
        message = response['choices'][0]['message']
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

    @override
    def list_models(self) -> List[str]:
        headers = {
            'Accept': 'application/json',
            'Authorization': f'Bearer {self.settings.api_key.get_secret_value()}',
        }
        parameters: HttpGetKwargs = {
            'url': f'{self.settings.api_base}/models',
            'headers': headers,
        }
        response = self.http_client.get(parameters)
        self.http_client.raise_for_status(response)
        return [i['id'] for i in response.json()['data'] if i['object'] == 'model']

    def _parse_finish_reason(self, response: dict[str, Any]) -> FinishReason | None:
        choice = response['choices'][0]
        finish_reason = choice.get('finish_reason') or None
        if finish_reason is None:
            finish_reason: str | None = finish_details['type'] if (finish_details := choice.get('finish_details')) else None
        if finish_reason is not None:
            finish_reason = FinishReason(finish_reason)
        return finish_reason

    def _parse_usage(self, response: dict[str, Any]) -> Usage:
        if usage := response.get('usage'):
            input_tokens = usage['prompt_tokens']
            output_tokens = usage['completion_tokens']
            cost = self.cost(input_tokens, output_tokens)
            if cost is None:
                for k, v in usage.items():
                    if k in ('cost', 'total_cost'):
                        cost = v
                        break
            return Usage(input_tokens=input_tokens, output_tokens=output_tokens, cost=cost)
        return Usage()

    def _parse_extra_info(self, response: dict[str, Any]) -> dict[str, Any]:
        return {'response': response}

    def _update_delta(self, delta_dict: dict[str, Any], stream_manager: StreamManager) -> None:
        delta_content: str = delta_dict.get('content') or ''
        stream_manager.delta = delta_content

        if delta_dict.get('tool_calls'):
            index = delta_dict['tool_calls'][0]['index']
            if index >= len(stream_manager.tool_calls or []):
                new_tool_calls_message = parse_message_dict(delta_dict).tool_calls
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
