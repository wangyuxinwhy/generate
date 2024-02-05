from __future__ import annotations

import base64
import json
from typing import Any, AsyncIterator, ClassVar, Dict, Iterator, List, Literal, Optional, Union

from pydantic import field_validator
from typing_extensions import NotRequired, Self, TypedDict, Unpack, override

from generate.chat_completion.base import RemoteChatCompletionModel
from generate.chat_completion.message import (
    AssistantMessage,
    FunctionCall,
    ImagePart,
    ImageUrlPart,
    Message,
    Messages,
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
from generate.chat_completion.model_output import ChatCompletionOutput, ChatCompletionStreamOutput, Stream
from generate.http import (
    HttpClient,
    HttpxPostKwargs,
    ResponseValue,
    UnexpectedResponseError,
)
from generate.model import ModelInfo, ModelParameters, ModelParametersDict
from generate.platforms.zhipu import ZhipuSettings, generate_zhipu_token
from generate.types import JsonSchema, Probability, Temperature


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


class ZhipuChatParametersDict(ModelParametersDict, total=False):
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


def convert_to_zhipu_message(message: Message) -> ZhipuMessage:
    if isinstance(message, UserMessage):
        return {
            'role': 'user',
            'content': message.content,
        }

    if isinstance(message, UserMultiPartMessage):
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

    if isinstance(message, AssistantMessage):
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

    if isinstance(message, SystemMessage):
        return {
            'role': 'system',
            'content': message.content,
        }

    if isinstance(message, ToolMessage):
        return {
            'role': 'tool',
            'content': message.content or '',
            'tool_call_id': message.tool_call_id,
        }

    raise MessageTypeError(message, (UserMessage, AssistantMessage))


def _convert_to_assistant_message(zhiput_message_dict: dict[str, Any]) -> AssistantMessage:
    if 'tool_calls' in zhiput_message_dict:
        dict_format_tool_calls = zhiput_message_dict['tool_calls']
        dict_format_tool_calls.sort(key=lambda x: x['index'])
        tool_calls = []
        for tool_call_dict in zhiput_message_dict['tool_calls']:
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
            content='',
            tool_calls=tool_calls,
        )
    return AssistantMessage(
        role='assistant',
        content=zhiput_message_dict['content'],
    )


def _calculate_cost(model_name: str, usage: dict[str, Any]) -> float | None:
    if model_name == 'glm-4':
        return 0.1 * (usage['total_tokens'] / 1000)
    if model_name == 'glm-3-turbo':
        return 0.005 * (usage['total_tokens'] / 1000)
    if model_name == 'characterglm':
        return 0.015 * (usage['total_tokens'] / 1000)
    return None


class _StreamResponseProcessor:
    def __init__(self) -> None:
        self.message: AssistantMessage | None = None
        self.is_start = True

    def process(self, stream_line: str) -> ChatCompletionStreamOutput | None:
        if not stream_line.strip():
            return None

        line = self._preprocess_stream_line(stream_line)
        if not line:
            return None
        response = json.loads(line)
        delta_dict = response['choices'][0]['delta']

        if self.message is None:
            if self._is_contains_content(delta_dict):
                self.message = self.process_initial_message(delta_dict)
            else:
                return None
        else:
            self.update_existing_message(delta_dict)
        extra = self.extract_extra_info(response)
        cost = cost = self.calculate_response_cost(response)
        finish_reason = self.determine_finish_reason(response)
        stream_control = 'finish' if finish_reason else 'start' if self.is_start else 'continue'
        self.is_start = False
        return ChatCompletionStreamOutput(
            model_info=ModelInfo(task='chat_completion', type='zhipu', name=response['model']),
            message=self.message,
            finish_reason=finish_reason,
            cost=cost,
            extra=extra,
            stream=Stream(delta=delta_dict.get('content') or '', control=stream_control),
        )

    @staticmethod
    def _preprocess_stream_line(line: str) -> str:
        line = line.replace('data:', '')
        return line.strip()

    def _is_contains_content(self, delta_dict: dict[str, Any]) -> bool:
        return not (
            delta_dict.get('content') is None
            and delta_dict.get('tool_calls') is None
            and delta_dict.get('function_call') is None
        )

    def process_initial_message(self, delta_dict: dict[str, Any]) -> AssistantMessage:
        return _convert_to_assistant_message(delta_dict)

    def update_existing_message(self, delta_dict: dict[str, Any]) -> None:
        if not delta_dict:
            return
        assert self.message is not None

        delta_content = delta_dict.get('content', '')
        self.message.content += delta_content

        if delta_dict.get('tool_calls'):
            index = delta_dict['tool_calls'][0]['index']
            if index >= len(self.message.tool_calls or []):
                new_tool_calls_message = _convert_to_assistant_message(delta_dict).tool_calls
                assert new_tool_calls_message is not None
                if self.message.tool_calls is None:
                    self.message.tool_calls = []
                self.message.tool_calls.append(new_tool_calls_message[0])
            else:
                assert self.message.tool_calls is not None
                self.message.tool_calls[index].function.arguments += delta_dict['tool_calls'][0]['function']['arguments']

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
            return _calculate_cost(response['model'], usage)
        return None

    def determine_finish_reason(self, response: ResponseValue) -> str | None:
        return response['choices'][0].get('finish_reason')


class ZhipuChat(RemoteChatCompletionModel):
    model_type: ClassVar[str] = 'zhipu'

    parameters: ZhipuChatParameters
    settings: ZhipuSettings

    def __init__(
        self,
        model: str = 'glm-4',
        parameters: ZhipuChatParameters | None = None,
        settings: ZhipuSettings | None = None,
        http_client: HttpClient | None = None,
    ) -> None:
        parameters = parameters or ZhipuChatParameters()
        settings = settings or ZhipuSettings()  # type: ignore
        http_client = http_client or HttpClient(stream_strategy='basic')
        super().__init__(parameters=parameters, settings=settings, http_client=http_client)

        self.model = model

    def _get_request_parameters(self, messages: Messages, parameters: ZhipuChatParameters) -> HttpxPostKwargs:
        zhipu_messages = self._convert_messages(messages)
        headers = {
            'Authorization': generate_zhipu_token(self.settings.api_key.get_secret_value()),
        }
        params = {'messages': zhipu_messages, 'model': self.model, **parameters.custom_model_dump()}
        return {
            'url': f'{self.settings.v4_api_base}/chat/completions',
            'headers': headers,
            'json': params,
        }

    def _parse_reponse(self, response: ResponseValue) -> ChatCompletionOutput:
        message_dict = response['choices'][0]['message']
        return ChatCompletionOutput(
            model_info=self.model_info,
            message=_convert_to_assistant_message(message_dict),
            cost=_calculate_cost(self.model, response['usage']),
            extra={'usage': response['usage']},
        )

    def _get_stream_request_parameters(self, messages: Messages, parameters: ZhipuChatParameters) -> HttpxPostKwargs:
        http_parameters = self._get_request_parameters(messages, parameters)
        http_parameters['json']['stream'] = True
        return http_parameters

    def _convert_messages(self, messages: Messages) -> list[ZhipuMessage]:
        return [convert_to_zhipu_message(message) for message in messages]

    @override
    def generate(self, prompt: Prompt, **kwargs: Unpack[ZhipuChatParametersDict]) -> ChatCompletionOutput:
        messages = ensure_messages(prompt)
        parameters = self.parameters.clone_with_changes(**kwargs)
        request_parameters = self._get_request_parameters(messages, parameters)
        response = self.http_client.post(request_parameters=request_parameters)
        return self._parse_reponse(response.json())

    @override
    async def async_generate(self, prompt: Prompt, **kwargs: Unpack[ZhipuChatParametersDict]) -> ChatCompletionOutput:
        messages = ensure_messages(prompt)
        parameters = self.parameters.clone_with_changes(**kwargs)
        request_parameters = self._get_request_parameters(messages, parameters)
        response = await self.http_client.async_post(request_parameters=request_parameters)
        return self._parse_reponse(response.json())

    @override
    def stream_generate(
        self, prompt: Prompt, **kwargs: Unpack[ZhipuChatParametersDict]
    ) -> Iterator[ChatCompletionStreamOutput]:
        messages = ensure_messages(prompt)
        parameters = self.parameters.clone_with_changes(**kwargs)
        request_parameters = self._get_stream_request_parameters(messages, parameters)
        stream_processor = _StreamResponseProcessor()
        is_finish = False
        for line in self.http_client.stream_post(request_parameters=request_parameters):
            if is_finish:
                continue
            output = stream_processor.process(line)
            if output is None:
                continue
            is_finish = output.is_finish
            yield output

    @override
    async def async_stream_generate(
        self, prompt: Prompt, **kwargs: Unpack[ZhipuChatParametersDict]
    ) -> AsyncIterator[ChatCompletionStreamOutput]:
        messages = ensure_messages(prompt)
        parameters = self.parameters.clone_with_changes(**kwargs)
        request_parameters = self._get_stream_request_parameters(messages, parameters)
        stream_processor = _StreamResponseProcessor()
        is_finish = False
        async for line in self.http_client.async_stream_post(request_parameters=request_parameters):
            if is_finish:
                continue
            output = stream_processor.process(line)
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


class ZhipuMeta(TypedDict):
    user_info: str
    bot_info: str
    bot_name: str
    user_name: str


class ZhipuCharacterChatParameters(ModelParameters):
    meta: ZhipuMeta = {
        'user_info': '我是陆星辰，是一个男性，是一位知名导演，也是苏梦远的合作导演。',
        'bot_info': '苏梦远，本名苏远心，是一位当红的国内女歌手及演员。',
        'bot_name': '苏梦远',
        'user_name': '陆星辰',
    }
    request_id: Optional[str] = None

    def custom_model_dump(self) -> dict[str, Any]:
        output = super().custom_model_dump()
        output['return_type'] = 'text'
        return output


class ZhipuCharacterChatParametersDict(ModelParametersDict, total=False):
    meta: ZhipuMeta
    request_id: Optional[str]


class ZhipuCharacterChat(RemoteChatCompletionModel):
    model_type: ClassVar[str] = 'zhipu_character'

    parameters: ZhipuCharacterChatParameters
    settings: ZhipuSettings

    def __init__(
        self,
        model: str = 'charglm-3',
        parameters: ZhipuCharacterChatParameters | None = None,
        settings: ZhipuSettings | None = None,
        http_client: HttpClient | None = None,
    ) -> None:
        parameters = parameters or ZhipuCharacterChatParameters()
        settings = settings or ZhipuSettings()  # type: ignore
        http_client = http_client or HttpClient()
        super().__init__(parameters=parameters, settings=settings, http_client=http_client)

        self.model = model

    def _get_request_parameters(self, messages: Messages, parameters: ModelParameters) -> HttpxPostKwargs:
        zhipu_messages = self._convert_messages(messages)
        headers = {
            'Authorization': generate_zhipu_token(self.settings.api_key.get_secret_value()),
        }
        params = {'prompt': zhipu_messages, **parameters.custom_model_dump()}
        return {
            'url': f'{self.settings.v3_api_base}/{self.model}/invoke',
            'headers': headers,
            'json': params,
        }

    def _convert_messages(self, messages: Messages) -> list[ZhipuMessage]:
        return [convert_to_zhipu_message(message) for message in messages]

    def _parse_reponse(self, response: ResponseValue) -> ChatCompletionOutput:
        if response['success']:
            text = response['data']['choices'][0]['content']
            return ChatCompletionOutput(
                model_info=self.model_info,
                message=AssistantMessage(content=text),
                cost=_calculate_cost(self.name, response['data']['usage']),
                extra={'usage': response['data']['usage']},
            )

        raise UnexpectedResponseError(response)

    def _get_stream_request_parameters(self, messages: Messages, parameters: ModelParameters) -> HttpxPostKwargs:
        http_parameters = self._get_request_parameters(messages, parameters)
        http_parameters['url'] = f'{self.settings.v3_api_base}/{self.model}/sse-invoke'
        return http_parameters

    @override
    def generate(self, prompt: Prompt, **kwargs: Unpack[ZhipuCharacterChatParametersDict]) -> ChatCompletionOutput:
        messages = ensure_messages(prompt)
        parameters = self.parameters.clone_with_changes(**kwargs)
        request_parameters = self._get_request_parameters(messages, parameters)
        response = self.http_client.post(request_parameters=request_parameters)
        return self._parse_reponse(response.json())

    @override
    async def async_generate(self, prompt: Prompt, **kwargs: Unpack[ZhipuCharacterChatParametersDict]) -> ChatCompletionOutput:
        messages = ensure_messages(prompt)
        parameters = self.parameters.clone_with_changes(**kwargs)
        request_parameters = self._get_request_parameters(messages, parameters)
        response = await self.http_client.async_post(request_parameters=request_parameters)
        return self._parse_reponse(response.json())

    @override
    def stream_generate(
        self, prompt: Prompt, **kwargs: Unpack[ZhipuCharacterChatParametersDict]
    ) -> Iterator[ChatCompletionStreamOutput]:
        messages = ensure_messages(prompt)
        parameters = self.parameters.clone_with_changes(**kwargs)
        request_parameters = self._get_stream_request_parameters(messages, parameters)
        message = AssistantMessage(content='')
        is_start = True
        for line in self.http_client.stream_post(request_parameters=request_parameters):
            message.content += line
            yield ChatCompletionStreamOutput(
                model_info=self.model_info,
                message=message,
                stream=Stream(delta=line, control='start' if is_start else 'continue'),
            )
            is_start = False
        yield ChatCompletionStreamOutput(
            model_info=self.model_info,
            message=message,
            finish_reason='stop',
            stream=Stream(delta='', control='finish'),
        )

    @override
    async def async_stream_generate(
        self, prompt: Prompt, **kwargs: Unpack[ZhipuCharacterChatParametersDict]
    ) -> AsyncIterator[ChatCompletionStreamOutput]:
        messages = ensure_messages(prompt)
        parameters = self.parameters.clone_with_changes(**kwargs)
        request_parameters = self._get_stream_request_parameters(messages, parameters)
        message = AssistantMessage(content='')
        is_start = True
        async for line in self.http_client.async_stream_post(request_parameters=request_parameters):
            message.content += line
            yield ChatCompletionStreamOutput(
                model_info=self.model_info,
                message=message,
                stream=Stream(delta=line, control='start' if is_start else 'continue'),
            )
            is_start = False
        yield ChatCompletionStreamOutput(
            model_info=self.model_info,
            message=message,
            finish_reason='stop',
            stream=Stream(delta='', control='finish'),
        )

    @property
    @override
    def name(self) -> str:
        return self.model

    @classmethod
    @override
    def from_name(cls, name: str) -> Self:
        return cls(model=name)
