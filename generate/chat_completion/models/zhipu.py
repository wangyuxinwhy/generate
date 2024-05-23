from __future__ import annotations

import base64
from typing import Any, AsyncIterator, ClassVar, Dict, Iterator, List, Literal, Optional, Union

from pydantic import field_validator
from typing_extensions import NotRequired, TypedDict, Unpack, override

from generate.chat_completion.base import RemoteChatCompletionModel
from generate.chat_completion.message import (
    AssistantMessage,
    FunctionCall,
    ImagePart,
    ImageUrlPart,
    Messages,
    Prompt,
    SystemMessage,
    TextPart,
    ToolCall,
    ToolMessage,
    UserMessage,
    UserMultiPartMessage,
)
from generate.chat_completion.message.converter import MessageConverter
from generate.chat_completion.message.core import FunctionMessage
from generate.chat_completion.message.exception import MessageTypeError
from generate.chat_completion.model_output import ChatCompletionOutput, ChatCompletionStreamOutput, FinishReason, Usage
from generate.chat_completion.stream_manager import StreamManager
from generate.chat_completion.tool import SupportToolCall, Tool
from generate.http import (
    HttpClient,
    HttpxPostKwargs,
    ResponseValue,
)
from generate.model import ModelParameters, RemoteModelParametersDict
from generate.platforms.zhipu import ZhipuSettings, generate_zhipu_token
from generate.types import JsonSchema, OrIterable, Probability, Temperature
from generate.utils import ensure_iterable

ZhipuModelPrice = {
    'glm-4v': (100, 100),
    'glm-4': (100, 100),
    'glm-3-turbo': (5, 5),
}


class Function(TypedDict):
    name: str
    description: str
    parameters: NotRequired[JsonSchema]


class Retrieval(TypedDict):
    knowledge_id: str
    prompt_template: NotRequired[str]


class WebSearch(TypedDict):
    enable: NotRequired[bool]
    search_query: NotRequired[str]


class ZhipuFunctionTool(TypedDict):
    type: Literal['function']
    function: Function


class ZhipuRetrievalTool(TypedDict):
    type: Literal['retrieval']
    retrieval: Retrieval


class ZhipuWebSearchTool(TypedDict):
    type: Literal['web_search']
    web_search: WebSearch


ZhipuTool = Union[ZhipuFunctionTool, ZhipuRetrievalTool, ZhipuWebSearchTool]


class ZhipuChatParameters(ModelParameters):
    temperature: Optional[Temperature] = None
    top_p: Optional[Probability] = None
    do_sample: Optional[bool] = None
    request_id: Optional[str] = None
    max_tokens: Optional[int] = None
    stop: Optional[List[str]] = None
    tools: Optional[List[ZhipuTool]] = None
    tool_choice: Optional[str] = None

    @field_validator('temperature')
    @classmethod
    def can_not_equal_zero(cls, v: Optional[Temperature]) -> Optional[Temperature]:
        if v == 0:
            return 0.01
        if v == 1:
            return 0.99
        return v


class ZhipuChatParametersDict(RemoteModelParametersDict, total=False):
    temperature: Optional[Temperature]
    top_p: Optional[Probability]
    request_id: Optional[str]
    max_tokens: Optional[int]
    stop: Optional[list[str]]
    tools: Optional[list[ZhipuTool]]
    tool_choice: Optional[str]


class ZhipuToolCall(TypedDict):
    id: str
    type: str
    index: int
    function: NotRequired[ZhipuFunctionCall]


class ZhipuFunctionCall(TypedDict):
    name: str
    arguments: str


class ZhipuMessage(TypedDict):
    role: Literal['user', 'assistant', 'system', 'tool']
    content: NotRequired[Union[str, List[Dict[str, str]]]]
    tool_calls: NotRequired[list[ZhipuToolCall]]
    tool_call_id: NotRequired[str]


class ZhipuMessageConverter(MessageConverter):
    allowed_message_types = [SystemMessage, UserMessage, AssistantMessage, ToolMessage, UserMultiPartMessage]

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

    def convert_assistant_message(self, message: AssistantMessage) -> Dict[str, Any]:
        if message.tool_calls is not None:
            dict_format_toll_calls: list[ZhipuToolCall] = []
            for index, tool_call in enumerate(message.tool_calls):
                tool_type = tool_call.type
                if tool_type not in {'function', 'retrieval', 'web_search'}:
                    raise ValueError(f'invalid tool type: {tool_type}, should be one of function, retrieval, web_search')
                dict_format_toll_call: ZhipuToolCall = {
                    'id': tool_call.id,
                    'type': tool_type,
                    'index': index,
                }
                if tool_type == 'function':
                    function_dict: ZhipuFunctionCall = {
                        'name': tool_call.function.name,
                        'arguments': tool_call.function.arguments,
                    }
                    dict_format_toll_call['function'] = function_dict
                dict_format_toll_calls.append(dict_format_toll_call)
            return {
                'role': 'assistant',
                'tool_calls': dict_format_toll_calls,
            }
        return {
            'role': 'assistant',
            'content': message.content,
        }

    def convert_tool_message(self, message: ToolMessage) -> Dict[str, Any]:
        return {
            'role': 'tool',
            'content': message.content or '',
            'tool_call_id': message.tool_call_id,
        }

    def convert_function_message(self, message: FunctionMessage) -> Dict[str, Any]:
        raise MessageTypeError(message, allowed_message_type=self.allowed_message_types)

    def convert_user_multi_part_message(self, message: UserMultiPartMessage) -> Dict[str, Any]:
        content = []
        for part in message.content:
            if isinstance(part, TextPart):
                content.append(
                    {
                        'type': 'text',
                        'text': part.text,
                    }
                )
            elif isinstance(part, ImageUrlPart):
                content.append(
                    {
                        'type': 'image_url',
                        'image_url': {
                            'url': part.image_url.url,
                        },
                    }
                )
            elif isinstance(part, ImagePart):
                content.append(
                    {
                        'type': 'image_url',
                        'image_url': {
                            'url': base64.b64encode(part.image).decode(),
                        },
                    }
                )
        return {'role': 'user', 'content': content}


class ZhipuChat(RemoteChatCompletionModel, SupportToolCall):
    model_type: ClassVar[str] = 'zhipu'
    available_models: ClassVar[List[str]] = ['glm-4', 'glm-3-turbo', 'glm-4v']

    parameters: ZhipuChatParameters
    settings: ZhipuSettings

    def __init__(
        self,
        model: str = 'glm-4',
        parameters: ZhipuChatParameters | None = None,
        settings: ZhipuSettings | None = None,
        http_client: HttpClient | None = None,
        message_converter: ZhipuMessageConverter | None = None,
    ) -> None:
        parameters = parameters or ZhipuChatParameters()
        settings = settings or ZhipuSettings()  # type: ignore
        http_client = http_client or HttpClient()
        message_converter = message_converter or ZhipuMessageConverter()
        super().__init__(
            model=model,
            parameters=parameters,
            settings=settings,
            http_client=http_client,
            message_converter=message_converter,
        )

    @override
    def generate(self, prompt: Prompt, **kwargs: Unpack[ZhipuChatParametersDict]) -> ChatCompletionOutput:
        return super().generate(prompt, **kwargs)

    @override
    async def async_generate(self, prompt: Prompt, **kwargs: Unpack[ZhipuChatParametersDict]) -> ChatCompletionOutput:
        return await super().async_generate(prompt, **kwargs)

    @override
    def stream_generate(
        self, prompt: Prompt, **kwargs: Unpack[ZhipuChatParametersDict]
    ) -> Iterator[ChatCompletionStreamOutput]:
        yield from super().stream_generate(prompt, **kwargs)

    @override
    async def async_stream_generate(
        self, prompt: Prompt, **kwargs: Unpack[ZhipuChatParametersDict]
    ) -> AsyncIterator[ChatCompletionStreamOutput]:
        async for stream_output in super().async_stream_generate(prompt, **kwargs):
            yield stream_output

    @override
    def _get_request_parameters(
        self, messages: Messages, stream: bool = False, **kwargs: Unpack[ZhipuChatParametersDict]
    ) -> HttpxPostKwargs:
        parameters = self.parameters.clone_with_changes(**kwargs)
        headers = {
            'Authorization': generate_zhipu_token(self.settings.api_key.get_secret_value()),
        }
        params = {
            'messages': self.message_converter.convert_messages(messages),
            'model': self.model,
            'stream': stream,
            **parameters.custom_model_dump(),
        }
        return {
            'url': self.settings.api_base + '/chat/completions',
            'headers': headers,
            'json': params,
        }

    @override
    def _process_reponse(self, response: ResponseValue) -> ChatCompletionOutput:
        return ChatCompletionOutput(
            model_info=self.model_info,
            message=self._parse_assistant_message(response['choices'][0]['message']),
            usage=self._parse_usage(response),
            extra=self._parse_extra(response),
            finish_reason=self._parse_finish_reason(response),
        )

    @override
    def _process_stream_response(
        self, response: Dict[str, Any], stream_manager: StreamManager
    ) -> ChatCompletionStreamOutput | None:
        delta_dict = response['choices'][0]['delta']
        self._update_delta(delta_dict, stream_manager)
        stream_manager.finish_reason = self._parse_finish_reason(response)
        stream_manager.extra = self._parse_extra(response)
        stream_manager.usage = self._parse_usage(response)
        return stream_manager.build_stream_output()

    def add_tools(self, tools: OrIterable[Tool]) -> None:
        new_tools: list[ZhipuTool] = [
            {
                'type': 'function',
                'function': {
                    'name': tool.name,
                    'description': tool.description,
                    'parameters': tool.parameters,
                },
            }
            for tool in ensure_iterable(tools)
        ]
        if self.parameters.tools is None:
            self.parameters.tools = new_tools
        else:
            self.parameters.tools.extend(new_tools)

    def _parse_assistant_message(self, message: dict[str, Any]) -> AssistantMessage:
        if 'tool_calls' in message:
            dict_format_tool_calls = message['tool_calls']
            dict_format_tool_calls.sort(key=lambda x: x['index'])
            tool_calls = []
            for tool_call_dict in message['tool_calls']:
                if tool_call_dict['type'] != 'function':
                    raise ValueError(f'invalid tool type: {tool_call_dict["type"]}, should be function')
                tool_calls.append(
                    ToolCall(
                        id=tool_call_dict['id'],
                        type='function',
                        function=FunctionCall(
                            name=tool_call_dict['function']['name'],
                            arguments=tool_call_dict['function']['arguments'],
                        ),
                    )
                )
            return AssistantMessage(
                role='assistant',
                content=message.get('content') or '',
                tool_calls=tool_calls,
            )
        return AssistantMessage(
            role='assistant',
            content=message['content'],
        )

    def _parse_usage(self, response: dict[str, Any]) -> Usage:
        usage = response.get('usage')
        if usage is not None:
            input_tokens = usage['prompt_tokens']
            output_tokens = usage['completion_tokens']
            return Usage(input_tokens=input_tokens, output_tokens=output_tokens)
        return Usage()

    def _parse_finish_reason(self, response: dict[str, Any]) -> FinishReason | None:
        try:
            choice = response['choices'][0]
            if finish_reason := choice.get('finish_reason'):
                return FinishReason(finish_reason)
        except (KeyError, IndexError, ValueError):
            return None

    def _parse_extra(self, response: dict[str, Any]) -> dict[str, Any]:
        return {'response': response}

    def _update_delta(self, delta_dict: dict[str, Any], stream_manager: StreamManager) -> None:
        delta_content: str = delta_dict.get('content') or ''
        stream_manager.delta = delta_content

        if delta_dict.get('tool_calls'):
            index = delta_dict['tool_calls'][0]['index']
            if index >= len(stream_manager.tool_calls or []):
                new_tool_calls_message = self._parse_assistant_message(delta_dict).tool_calls
                assert new_tool_calls_message is not None
                if stream_manager.tool_calls is None:
                    stream_manager.tool_calls = []
                stream_manager.tool_calls.append(new_tool_calls_message[0])
            else:
                assert stream_manager.tool_calls is not None
                stream_manager.tool_calls[index].function.arguments += delta_dict['tool_calls'][0]['function']['arguments']
