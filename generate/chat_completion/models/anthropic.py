from __future__ import annotations

import base64
import json
import uuid
from typing import Any, AsyncIterator, ClassVar, Dict, Iterator, List, Literal, Optional

from pydantic import Field, PositiveInt
from typing_extensions import Annotated, NotRequired, TypedDict, Unpack, override

from generate.chat_completion.base import RemoteChatCompletionModel
from generate.chat_completion.message import Prompt
from generate.chat_completion.message.converter import MessageConverter
from generate.chat_completion.message.core import (
    AssistantMessage,
    FunctionCall,
    FunctionMessage,
    ImagePart,
    ImageUrlPart,
    Messages,
    SystemMessage,
    TextPart,
    ToolCall,
    ToolMessage,
    UserMessage,
    UserMultiPartMessage,
)
from generate.chat_completion.message.exception import MessageTypeError
from generate.chat_completion.model_output import ChatCompletionOutput, ChatCompletionStreamOutput, FinishReason, Usage
from generate.chat_completion.stream_manager import StreamManager
from generate.chat_completion.tool import SupportToolCall, Tool
from generate.http import HttpClient, HttpxPostKwargs
from generate.model import ModelParameters, RemoteModelParametersDict
from generate.platforms import AnthropicSettings
from generate.types import OrIterable, Probability, Temperature
from generate.utils import ensure_iterable


class AnthropicTool(TypedDict):
    name: str
    description: Optional[str]
    input_schema: Dict[str, Any]


class AnthropicToolChoice(TypedDict):
    type: Literal['auto', 'any', 'tool']
    name: NotRequired[str]


class AnthropicChatParameters(ModelParameters):
    system: Optional[str] = None
    max_tokens: PositiveInt = 1024
    metadata: Optional[Dict[str, Any]] = {}
    stop: Annotated[Optional[List[str]], Field(alias='stop_sequences')] = None
    temperature: Optional[Temperature] = None
    top_p: Optional[Probability] = None
    top_k: Optional[PositiveInt] = None
    tools: Optional[List[AnthropicTool]] = None
    tool_choice: Optional[AnthropicToolChoice] = None


class AnthropicParametersDict(RemoteModelParametersDict, total=False):
    system: Optional[str]
    max_tokens: PositiveInt
    metadata: Optional[Dict[str, Any]]
    stop: Optional[List[str]]
    temperature: Optional[Temperature]
    top_p: Optional[Probability]
    top_k: Optional[PositiveInt]
    tools: Optional[List[AnthropicTool]]
    tool_choice: Optional[AnthropicToolChoice]


class AnthropicMessageConverter(MessageConverter):
    allowed_message_types = [UserMessage, AssistantMessage, UserMultiPartMessage, ToolMessage]

    def __init__(self, http_client: HttpClient) -> None:
        super().__init__()
        self.http_client = http_client
        self.handle_tool_choice = True

    def convert_user_message(self, message: UserMessage) -> Dict[str, Any]:
        return {'role': 'user', 'content': message.content}

    def convert_assistant_message(self, message: AssistantMessage) -> Dict[str, Any]:
        content = []
        if message.content:
            content.append({'type': 'text', 'text': message.content})
        if message.tool_calls:
            for tool_call in message.tool_calls:
                content.append(
                    {
                        'type': 'tool_use',
                        'id': tool_call.id,
                        'name': tool_call.function.name,
                        'input': json.loads(tool_call.function.arguments),
                    }
                )
        return {'role': 'assistant', 'content': content}

    def convert_user_multi_part_message(self, message: UserMultiPartMessage) -> Dict[str, Any]:
        message_dict = {'role': 'user', 'content': []}
        for part in message.content:
            if isinstance(part, TextPart):
                message_dict['content'].append({'type': 'text', 'text': part.text})

            if isinstance(part, ImagePart):
                data = base64.b64encode(part.image).decode()
                media_type = 'image/jpeg' if part.image_format is None else f'image/{part.image_format}'
                message_dict['content'].append(
                    {'type': 'image', 'source': {'type': 'base64', 'media_type': media_type, 'data': data}}
                )

            if isinstance(part, ImageUrlPart):
                response = self.http_client.get({'url': part.image_url.url})
                data = base64.b64encode(response.content).decode()
                media_type = response.headers.get('Content-Type') or 'image/jpeg'
                message_dict['content'].append(
                    {'type': 'image', 'source': {'type': 'base64', 'media_type': media_type, 'data': data}}
                )
        return message_dict

    def convert_system_message(self, message: SystemMessage) -> Dict[str, Any]:
        raise MessageTypeError(message, self.allowed_message_types)

    def convert_function_message(self, message: FunctionMessage) -> Dict[str, Any]:
        raise MessageTypeError(message, self.allowed_message_types)

    def convert_tool_message(self, message: ToolMessage) -> Dict[str, Any]:
        tool_result: dict = {
            'type': 'tool_result',
            'tool_use_id': message.tool_call_id,
        }
        if message.content:
            tool_result['content'] = message.content
        if message.is_error:
            tool_result['is_error'] = True
        return {
            'role': 'user',
            'content': [tool_result],
        }


class AnthropicChat(RemoteChatCompletionModel, SupportToolCall):
    model_type: ClassVar[str] = 'anthropic'
    tools_beta_version: ClassVar[str] = 'tools-2024-05-16'
    available_models: ClassVar[List[str]] = [
        'claude-2.1',
        'claude-2.0',
        'claude-instant-1.2',
        'claude-3-opus-20240229',
        'claude-3-sonnet-20240229',
        'claude-3-haiku-20240307',
    ]

    parameters: AnthropicChatParameters
    settings: AnthropicSettings
    message_converter: AnthropicMessageConverter

    def __init__(
        self,
        model: str = 'claude-3-haiku-20240307',
        parameters: AnthropicChatParameters | None = None,
        settings: AnthropicSettings | None = None,
        http_client: HttpClient | None = None,
        message_converter: AnthropicMessageConverter | None = None,
    ) -> None:
        parameters = parameters or AnthropicChatParameters()
        settings = settings or AnthropicSettings()  # type: ignore
        http_client = http_client or HttpClient()
        message_converter = message_converter or AnthropicMessageConverter(http_client)
        super().__init__(
            model=model,
            parameters=parameters,
            settings=settings,
            http_client=http_client,
            message_converter=message_converter,
        )

    @override
    def generate(self, prompt: Prompt, **kwargs: Unpack[AnthropicParametersDict]) -> ChatCompletionOutput:
        return super().generate(prompt, **kwargs)

    @override
    async def async_generate(self, prompt: Prompt, **kwargs: Unpack[AnthropicParametersDict]) -> ChatCompletionOutput:
        return await super().async_generate(prompt, **kwargs)

    @override
    def stream_generate(
        self, prompt: Prompt, **kwargs: Unpack[AnthropicParametersDict]
    ) -> Iterator[ChatCompletionStreamOutput]:
        yield from super().stream_generate(prompt, **kwargs)

    @override
    async def async_stream_generate(
        self, prompt: Prompt, **kwargs: Unpack[AnthropicParametersDict]
    ) -> AsyncIterator[ChatCompletionStreamOutput]:
        async for output in super().async_stream_generate(prompt, **kwargs):
            yield output

    @override
    def _get_request_parameters(
        self, messages: Messages, stream: bool = False, **kwargs: Unpack[AnthropicParametersDict]
    ) -> HttpxPostKwargs:
        parameters = self.parameters.clone_with_changes(**kwargs)
        if isinstance(messages[0], SystemMessage):
            parameters.system = messages[0].content
            messages = messages[1:]
        anthropic_messages = self.message_converter.convert_messages(messages)
        headers = {
            'Content-Type': 'application/json',
            'anthropic-version': self.settings.api_version,
            'x-api-key': self.settings.api_key.get_secret_value(),
        }
        if tool_use := bool(parameters.tools):
            headers['anthropic-beta'] = self.tools_beta_version

        json_dict = parameters.custom_model_dump()
        json_dict['model'] = self.model
        json_dict['messages'] = anthropic_messages

        if stream:
            if tool_use:
                raise ValueError('Tool calls are not supported in stream mode')
            json_dict['stream'] = True

        return {
            'url': self.settings.api_base + '/messages',
            'headers': headers,
            'json': json_dict,
        }

    @override
    def _process_reponse(self, response: dict[str, Any]) -> ChatCompletionOutput:
        return ChatCompletionOutput(
            model_info=self.model_info,
            message=self._parse_assistant_message(response),
            finish_reason=self._parse_finish_reason(response),
            usage=self._parse_usage(response),
            extra=self._parse_extra(response),
        )

    @override
    def _process_stream_response(
        self, response: dict[str, Any], stream_manager: StreamManager
    ) -> ChatCompletionStreamOutput | None:
        if 'message' in response:
            input_tokens = response['message']['usage']['input_tokens']
            stream_manager.usage.input_tokens = input_tokens
            return None

        if 'delta' in response:
            if 'stop_reason' in response['delta']:
                delta_dict = response['delta']
                stream_manager.delta = ''
                stream_manager.finish_reason = self._parse_finish_reason(delta_dict)
                stream_manager.usage.output_tokens = response['usage']['output_tokens']
                return stream_manager.build_stream_output()

            stream_manager.delta = response['delta']['text']
            return stream_manager.build_stream_output()
        return None

    @override
    def add_tools(self, tools: OrIterable[Tool]) -> None:
        new_tools = [
            AnthropicTool(name=tool.name, description=tool.description, input_schema=tool.parameters)
            for tool in ensure_iterable(tools)
        ]
        if self.parameters.tools is None:
            self.parameters.tools = new_tools
        else:
            self.parameters.tools.extend(new_tools)

    @override
    def generate_tool_call_id(self, function_call: FunctionCall) -> str:
        return f'toolu_{uuid.uuid4().hex}'

    def _parse_assistant_message(self, response: dict[str, Any]) -> AssistantMessage:
        content = ''
        tool_calls = []
        for i in response['content']:
            if i['type'] == 'text':
                content += i['text']
            if i['type'] == 'tool_use':
                tool_call = ToolCall(
                    id=i['id'], function=FunctionCall(name=i['name'], arguments=json.dumps(i['input'], ensure_ascii=False))
                )
                tool_calls.append(tool_call)
        tool_calls = tool_calls or None
        return AssistantMessage(content=content, tool_calls=tool_calls)

    def _parse_usage(self, response: dict[str, Any]) -> Usage:
        if 'usage' not in response:
            return Usage()

        input_tokens = response['usage']['input_tokens']
        output_tokens = response['usage']['output_tokens']
        return Usage(input_tokens=input_tokens, output_tokens=output_tokens)

    def _parse_finish_reason(self, response: dict[str, Any]) -> FinishReason | None:
        finish_reason_mapping = {
            'end_turn': 'end_turn',
            'max_tokens': 'length',
            'stop_sequence': 'stop',
            'tool_use': 'tool_calls',
        }
        finish_reason = finish_reason_mapping.get(response['stop_reason'])
        return FinishReason(finish_reason) if finish_reason else None

    def _parse_extra(self, response: dict[str, Any]) -> Dict[str, Any]:
        return {
            'response': response,
        }
