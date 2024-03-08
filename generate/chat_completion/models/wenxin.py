from __future__ import annotations

import json
from typing import Any, AsyncIterator, ClassVar, Dict, Iterator, List, Literal, Optional

from pydantic import Field, model_validator
from typing_extensions import Annotated, NotRequired, Self, TypedDict, Unpack, override

from generate.chat_completion.base import RemoteChatCompletionModel
from generate.chat_completion.message import (
    AssistantMessage,
    FunctionCall,
    FunctionMessage,
    Message,
    Messages,
    MessageTypeError,
    Prompt,
    SystemMessage,
    UserMessage,
    ensure_messages,
)
from generate.chat_completion.model_output import ChatCompletionOutput, ChatCompletionStreamOutput
from generate.chat_completion.stream_manager import StreamManager
from generate.chat_completion.tool import Tool, ToolCallMixin
from generate.http import (
    HttpClient,
    HttpxPostKwargs,
    ResponseValue,
    UnexpectedResponseError,
)
from generate.model import ModelParameters, RemoteModelParametersDict
from generate.platforms.baidu import QianfanSettings, QianfanTokenManager
from generate.types import JsonSchema, OrIterable, Probability, Temperature
from generate.utils import ensure_iterable


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


def convert_to_wenxin_function(tool: Tool) -> WenxinFunction:
    return {
        'name': tool.name,
        'description': tool.description,
        'parameters': tool.parameters,
    }


def convert_to_wenxin_message(message: Message) -> WenxinMessage:
    if isinstance(message, UserMessage):
        return {
            'role': 'user',
            'content': message.content,
        }

    if isinstance(message, AssistantMessage):
        if message.function_call:
            return {
                'role': 'assistant',
                'function_call': {
                    'name': message.function_call.name,
                    'arguments': message.function_call.arguments,
                    'thoughts': message.function_call.thoughts or '',
                },
                'content': message.content,
            }
        return {
            'role': 'assistant',
            'content': message.content,
        }

    if isinstance(message, FunctionMessage):
        return {
            'role': 'function',
            'name': message.name,
            'content': message.content,
        }

    raise MessageTypeError(message, allowed_message_type=(UserMessage, AssistantMessage, FunctionMessage))


def _convert_messages(messages: Messages) -> list[WenxinMessage]:
    if isinstance(system_message := messages[0], SystemMessage):
        prepend_messages = [UserMessage(content=system_message.content), AssistantMessage(content='好的')]
        messages = prepend_messages + messages[1:]
    return [convert_to_wenxin_message(message) for message in messages]


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


class WenxinChatParametersDict(RemoteModelParametersDict, total=False):
    temperature: Optional[Temperature]
    top_p: Optional[Probability]
    functions: Optional[List[WenxinFunction]]
    penalty_score: Optional[float]
    system: Optional[str]
    user: Optional[str]


class WenxinChat(RemoteChatCompletionModel, ToolCallMixin):
    model_type: ClassVar[str] = 'wenxin'
    model_name_entrypoint_map: ClassVar[Dict[str, str]] = {
        'ERNIE-Bot': 'completions',
        'ERNIE-Bot-turbo': 'eb-instant',
        'ERNIE-Bot-4': 'completions_pro',
    }
    available_models: ClassVar[List[str]] = ['ERNIE-Bot', 'ERNIE-Bot-turbo', 'ERNIE-Bot-4']

    parameters: WenxinChatParameters
    settings: QianfanSettings

    def __init__(
        self,
        model: str = 'ERNIE-Bot',
        parameters: WenxinChatParameters | None = None,
        settings: QianfanSettings | None = None,
        http_client: HttpClient | None = None,
    ) -> None:
        parameters = parameters or WenxinChatParameters()
        settings = settings or QianfanSettings()  # type: ignore
        http_client = http_client or HttpClient()
        super().__init__(model=model, parameters=parameters, settings=settings, http_client=http_client)
        self.token_manager = QianfanTokenManager(self.settings, self.http_client)

    @override
    def generate(self, prompt: Prompt, **kwargs: Unpack[WenxinChatParametersDict]) -> ChatCompletionOutput:
        return super().generate(prompt, **kwargs)

    @override
    async def async_generate(self, prompt: Prompt, **kwargs: Unpack[WenxinChatParametersDict]) -> ChatCompletionOutput:
        return await super().async_generate(prompt, **kwargs)

    @override
    def stream_generate(
        self, prompt: Prompt, **kwargs: Unpack[WenxinChatParametersDict]
    ) -> Iterator[ChatCompletionStreamOutput]:
        yield from super().stream_generate(prompt, **kwargs)

    @override
    async def async_stream_generate(
        self, prompt: Prompt, **kwargs: Unpack[WenxinChatParametersDict]
    ) -> AsyncIterator[ChatCompletionStreamOutput]:
        async for output in super().async_stream_generate(prompt, **kwargs):
            yield output

    def _get_request_parameters(
        self, prompt: Prompt, stream: bool = False, **kwargs: Unpack[WenxinChatParametersDict]
    ) -> HttpxPostKwargs:
        messages = ensure_messages(prompt)
        parameters = self.parameters.clone_with_changes(**kwargs)
        wenxin_messages: list[WenxinMessage] = _convert_messages(messages)
        parameters_dict = parameters.custom_model_dump()
        if 'temperature' in parameters_dict:
            parameters_dict['temperature'] = max(0.01, parameters_dict['temperature'])
        json_data = {'messages': wenxin_messages, **parameters_dict}
        if stream:
            json_data['stream'] = True

        return {
            'url': self.settings.comlpetion_api_base + self.model_name_entrypoint_map[self.model],
            'json': json_data,
            'params': {'access_token': self.token_manager.token},
            'headers': {'Content-Type': 'application/json'},
        }

    @override
    def _process_reponse(self, response: ResponseValue) -> ChatCompletionOutput:
        if response.get('error_msg'):
            raise UnexpectedResponseError(response)
        if response.get('function_call'):
            function_call = FunctionCall(
                name=response['function_call']['name'],
                arguments=response['function_call']['arguments'],
                thoughts=response['function_call']['thoughts'],
            )
        else:
            function_call = None
        message = AssistantMessage(content=response['result'], function_call=function_call)
        return ChatCompletionOutput(
            model_info=self.model_info,
            message=message,
            cost=self._calculate_cost(response['usage']),
            extra={
                'is_truncated': response['is_truncated'],
                'need_clear_history': response['need_clear_history'],
                'usage': response['usage'],
            },
            finish_reason=response['finish_reason'],
        )

    @override
    def _process_stream_line(self, line: str, stream_manager: StreamManager) -> ChatCompletionStreamOutput | None:
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            return None

        stream_manager.delta = data['result']
        if data['is_end']:
            stream_manager.cost = self._calculate_cost(data['usage'])
            stream_manager.finish_reason = 'stop'
            extra = {
                'is_truncated': data['is_truncated'],
                'need_clear_history': data['need_clear_history'],
                'usage': data['usage'],
            }
            stream_manager.extra.update(extra)
        return stream_manager.build_stream_output()

    def _calculate_cost(self, usage: dict[str, Any]) -> float | None:
        if self.name == 'ERNIE-Bot':
            return 0.012 * (usage['total_tokens'] / 1000)
        if self.name == 'ERNIE-Bot-turbo':
            return 0.008 * (usage['total_tokens'] / 1000)
        if self.name == 'ERNIE-Bot-4':
            return 0.12 * (usage['total_tokens'] / 1000)
        return None

    def add_tools(self, tools: OrIterable[Tool]) -> None:
        new_functions = [convert_to_wenxin_function(tool) for tool in ensure_iterable(tools)]
        if self.parameters.functions is None:
            self.parameters.functions = new_functions
        else:
            self.parameters.functions.extend(new_functions)
