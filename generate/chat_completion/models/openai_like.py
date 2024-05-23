from __future__ import annotations

import base64
from typing import Any, AsyncIterator, Dict, Iterator, List, Literal, Optional, Union, cast

from pydantic import Field, PositiveInt
from typing_extensions import Annotated, NotRequired, TypedDict, override

from generate.chat_completion.base import RemoteChatCompletionModel
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
from generate.chat_completion.message.core import Messages, Prompt
from generate.chat_completion.model_output import ChatCompletionOutput, ChatCompletionStreamOutput, FinishReason, Usage
from generate.chat_completion.stream_manager import StreamManager
from generate.chat_completion.tool import FunctionJsonSchema, SupportToolCall, Tool
from generate.http import (
    HttpClient,
    HttpGetKwargs,
    HttpxPostKwargs,
)
from generate.model import ModelInfo, ModelParameters, RemoteModelParametersDict
from generate.platforms.openai_like import OpenAILikeSettings
from generate.types import OrIterable, Probability, Temperature
from generate.utils import ensure_iterable


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
    logprobs: Optional[bool] = None
    top_logprobs: Optional[Annotated[int, Field(ge=0, le=20)]] = None
    user: Optional[str] = None
    response_format: Optional[OpenAIResponseFormat] = None
    seed: Optional[int] = None
    tools: Optional[List[OpenAITool]] = None
    tool_choice: Union[Literal['auto', 'none'], OpenAIToolChoice, None] = None


class OpenAIChatParametersDict(RemoteModelParametersDict, total=False):
    temperature: Optional[Temperature]
    top_p: Optional[Probability]
    max_tokens: Optional[PositiveInt]
    functions: Optional[List[FunctionJsonSchema]]
    function_call: Union[Literal['auto'], FunctionCallName, None]
    stop: Union[str, List[str], None]
    presence_penalty: Optional[float]
    frequency_penalty: Optional[float]
    logit_bias: Optional[Dict[int, int]]
    logprobs: Optional[bool]
    top_logprobs: Optional[int]
    user: Optional[str]
    response_format: Optional[OpenAIResponseFormat]
    seed: Optional[int]
    tools: Optional[List[OpenAITool]]
    tool_choice: Union[Literal['auto'], OpenAIToolChoice, None]


class SupportOpenAIToolCall(SupportToolCall):
    parameters: ModelParameters

    @override
    def add_tools(self, tools: OrIterable[Tool]) -> None:
        new_tools = [OpenAITool(type='function', function=tool.json_schema) for tool in ensure_iterable(tools)]
        if not hasattr(self.parameters, 'tools'):
            raise ValueError('The parameters must have a tools attribute')
        self.parameters = cast(OpenAIChatParameters, self.parameters)
        if self.parameters.tools is None:
            self.parameters.tools = new_tools
        else:
            self.parameters.tools.extend(new_tools)


class OpenAIMessageConverter(MessageConverter):
    allowed_message_types = [SystemMessage, UserMessage, UserMultiPartMessage, ToolMessage, AssistantMessage, FunctionMessage]

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


class OpenAILikeChat(RemoteChatCompletionModel):
    settings: OpenAILikeSettings

    message_converter: OpenAIMessageConverter
    parameters: ModelParameters
    settings: OpenAILikeSettings

    def __init__(
        self,
        model: str,
        parameters: ModelParameters | None = None,
        settings: OpenAILikeSettings | None = None,
        http_client: HttpClient | None = None,
        message_converter: MessageConverter | None = None,
    ) -> None:
        http_client = http_client or HttpClient()
        message_converter = message_converter or OpenAIMessageConverter()
        parameters = parameters or OpenAIChatParameters()
        if settings is None:
            raise ValueError('settings is required')
        super().__init__(
            model=model,
            parameters=parameters,
            settings=settings,
            http_client=http_client,
            message_converter=message_converter,
        )

    @override
    def generate(self, prompt: Prompt, **kwargs: Any) -> ChatCompletionOutput:
        return super().generate(prompt, **kwargs)

    @override
    async def async_generate(self, prompt: Prompt, **kwargs: Any) -> ChatCompletionOutput:
        return await super().async_generate(prompt, **kwargs)

    @override
    def stream_generate(self, prompt: Prompt, **kwargs: Any) -> Iterator[ChatCompletionStreamOutput]:
        yield from super().stream_generate(prompt, **kwargs)

    @override
    async def async_stream_generate(self, prompt: Prompt, **kwargs: Any) -> AsyncIterator[ChatCompletionStreamOutput]:
        async for stream_output in super().async_stream_generate(prompt, **kwargs):
            yield stream_output

    @override
    def _get_request_parameters(self, messages: Messages, stream: bool = False, **kwargs: Any) -> HttpxPostKwargs:
        parameters = self.parameters.clone_with_changes(**kwargs)
        headers = {
            'Authorization': f'Bearer {self.settings.api_key.get_secret_value()}',
        }
        json_data = {
            'model': self.model,
            'messages': self.message_converter.convert_messages(messages),
            **parameters.custom_model_dump(),
        }
        if stream:
            json_data['stream'] = True

        return {
            'url': f'{self.settings.api_base}/chat/completions',
            'headers': headers,
            'json': json_data,
        }

    @override
    def _process_reponse(self, response: dict[str, Any]) -> ChatCompletionOutput:
        return ChatCompletionOutput(
            model_info=ModelInfo(task='chat_completion', type=self.model_type, name=response['model']),
            message=self._parse_assistant_message(response['choices'][0]['message']),
            finish_reason=self._parse_finish_reason(response),
            usage=self._parse_usage(response),
            extra=self._parse_extra(response),
        )

    @override
    def _process_stream_response(
        self, response: Dict[str, Any], stream_manager: StreamManager
    ) -> ChatCompletionStreamOutput | None:
        delta_dict = response['choices'][0].get('delta', {})
        self._update_delta(delta_dict, stream_manager=stream_manager)
        stream_manager.extra = self._parse_extra(response)
        stream_manager.usage = self._parse_usage(response)
        stream_manager.finish_reason = self._parse_finish_reason(response)
        return stream_manager.build_stream_output()

    def _parse_assistant_message(self, message: dict[str, Any]) -> AssistantMessage:
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
            return Usage(input_tokens=input_tokens, output_tokens=output_tokens)
        return Usage()

    def _parse_extra(self, response: dict[str, Any]) -> dict[str, Any]:
        return {'response': response}

    def _update_delta(self, delta_dict: dict[str, Any], stream_manager: StreamManager) -> None:
        delta_content: str = delta_dict.get('content') or ''
        stream_manager.delta = delta_content

        if delta_dict.get('tool_calls'):
            tool_calls = delta_dict['tool_calls'][0]
            index = tool_calls['index']
            if index >= len(stream_manager.tool_calls or []):
                new_tool_calls_message = self._parse_assistant_message(delta_dict).tool_calls
                if new_tool_calls_message:
                    stream_manager.tool_calls.append(new_tool_calls_message[0])
            else:
                stream_manager.tool_calls[index].function.arguments += tool_calls['function']['arguments']

        if delta_dict.get('function_call'):
            if stream_manager.function_call is None:
                stream_manager.function_call = FunctionCall(name='', arguments='')
            function_name = delta_dict['function_call'].get('name', '')
            stream_manager.function_call.name += function_name
            arguments = delta_dict['function_call'].get('arguments', '')
            stream_manager.function_call.arguments += arguments
