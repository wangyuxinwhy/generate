from __future__ import annotations

from typing import Any, AsyncIterator, ClassVar, Iterator, List, Literal, Optional

from pydantic import Field
from typing_extensions import Annotated, NotRequired, TypedDict, Unpack, override

from generate.chat_completion.base import RemoteChatCompletionModel
from generate.chat_completion.message import (
    AssistantMessage,
    Messages,
    Prompt,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from generate.chat_completion.message.converter import MessageConverter, SimpleMessageConverter
from generate.chat_completion.message.core import FunctionCall, FunctionMessage, ToolCall, UserMultiPartMessage
from generate.chat_completion.message.exception import MessageTypeError
from generate.chat_completion.model_output import ChatCompletionOutput, ChatCompletionStreamOutput, FinishReason, Usage
from generate.chat_completion.models.openai_like import SupportOpenAIToolCall
from generate.chat_completion.stream_manager import StreamManager
from generate.chat_completion.tool import FunctionJsonSchema
from generate.http import (
    HttpClient,
    HttpxPostKwargs,
)
from generate.model import ModelParameters, RemoteModelParametersDict
from generate.platforms.baichuan import BaichuanSettings
from generate.types import Probability, Temperature


class BaichuanResponseFormat(TypedDict):
    type: Literal['json_object']


class BaichuanRetrieval(TypedDict):
    kb_ids: List[str]
    answer_model: NotRequired[str]


class BaichuanWebSearch(TypedDict):
    enable: NotRequired[bool]
    search_mode: NotRequired[str]


class BaichuanTool(TypedDict):
    type: Literal['retrieval', 'web_search', 'function']
    retrieval: NotRequired[BaichuanRetrieval]
    web_search: NotRequired[BaichuanWebSearch]
    function: NotRequired[FunctionJsonSchema]


class BaichuanChatParameters(ModelParameters):
    temperature: Optional[Temperature] = None
    top_k: Optional[Annotated[int, Field(ge=0)]] = None
    top_p: Optional[Probability] = None
    max_tokens: Optional[Annotated[int, Field(ge=0)]] = None
    response_format: Optional[BaichuanResponseFormat] = None
    tools: Optional[List[BaichuanTool]] = None
    tool_choice: Optional[str] = None


class BaichuanChatParametersDict(RemoteModelParametersDict, total=False):
    temperature: Optional[Temperature]
    top_k: Optional[int]
    top_p: Optional[Probability]
    max_tokens: Optional[int]
    response_format: Optional[BaichuanResponseFormat]
    tools: Optional[List[BaichuanTool]]
    tool_choice: Optional[str]


class BaichuanMessageConverter(MessageConverter):
    allowed_message_types = [SystemMessage, UserMessage, AssistantMessage, ToolMessage]

    def convert_system_message(self, message: SystemMessage) -> dict[str, Any]:
        return {
            'role': 'system',
            'content': message.content,
        }

    def convert_user_message(self, message: UserMessage) -> dict[str, Any]:
        return {
            'role': 'user',
            'content': message.content,
        }

    def convert_tool_message(self, message: ToolMessage) -> dict[str, Any]:
        return {
            'role': 'tool',
            'tool_call_id': message.tool_call_id,
            'content': message.content,
        }

    def convert_assistant_message(self, message: AssistantMessage) -> dict[str, Any]:
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
            raise ValueError('Function calls are not supported in Baichuan')
        return base_dict

    def convert_user_multi_part_message(self, message: UserMultiPartMessage) -> dict[str, Any]:
        raise MessageTypeError(message, allowed_message_type=self.allowed_message_types)

    def convert_function_message(self, message: FunctionMessage) -> dict[str, Any]:
        raise MessageTypeError(message, allowed_message_type=self.allowed_message_types)


class BaichuanChat(RemoteChatCompletionModel, SupportOpenAIToolCall):
    model_type: ClassVar[str] = 'baichuan'
    available_models: ClassVar[List[str]] = [
        'Baichuan2-Turbo',
        'Baichuan2-Turbo-192k',
        'Baichuan3-Turbo',
        'Baichuan3-Turbo-128k',
        'Baichuan4',
    ]

    parameters: BaichuanChatParameters
    settings: BaichuanSettings
    message_converter: SimpleMessageConverter

    def __init__(
        self,
        model: str = 'Baichuan3-Turbo',
        parameters: BaichuanChatParameters | None = None,
        settings: BaichuanSettings | None = None,
        http_client: HttpClient | None = None,
        message_converter: MessageConverter | None = None,
    ) -> None:
        parameters = parameters or BaichuanChatParameters()
        settings = settings or BaichuanSettings()  # type: ignore
        http_client = http_client or HttpClient()
        message_converter = message_converter or BaichuanMessageConverter()
        super().__init__(
            model=model,
            parameters=parameters,
            settings=settings,
            http_client=http_client,
            message_converter=message_converter,
        )

    @override
    def generate(self, prompt: Prompt, **kwargs: Unpack[BaichuanChatParametersDict]) -> ChatCompletionOutput:
        return super().generate(prompt, **kwargs)

    @override
    async def async_generate(self, prompt: Prompt, **kwargs: Unpack[BaichuanChatParametersDict]) -> ChatCompletionOutput:
        return await super().async_generate(prompt, **kwargs)

    @override
    def stream_generate(
        self, prompt: Prompt, **kwargs: Unpack[BaichuanChatParametersDict]
    ) -> Iterator[ChatCompletionStreamOutput]:
        yield from super().stream_generate(prompt, **kwargs)

    @override
    async def async_stream_generate(
        self, prompt: Prompt, **kwargs: Unpack[BaichuanChatParametersDict]
    ) -> AsyncIterator[ChatCompletionStreamOutput]:
        async for output in super().async_stream_generate(prompt, **kwargs):
            yield output

    @override
    def _get_request_parameters(
        self, messages: Messages, stream: bool = False, **kwargs: Unpack[BaichuanChatParametersDict]
    ) -> HttpxPostKwargs:
        if isinstance(system_message := messages[0], SystemMessage):
            prepend_messages = [UserMessage(content=system_message.content)]
            messages = prepend_messages + messages[1:]
        parameters = self.parameters.clone_with_changes(**kwargs)
        json_data = {
            'model': self.model,
            'messages': self.message_converter.convert_messages(messages),
        }
        parameters_dict = parameters.custom_model_dump()
        json_data.update(parameters_dict)
        if stream:
            json_data['stream'] = True
        headers = {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer ' + self.settings.api_key.get_secret_value(),
        }
        return {
            'url': self.settings.api_base + '/chat/completions',
            'headers': headers,
            'json': json_data,
        }

    @override
    def _process_reponse(self, response: dict[str, Any]) -> ChatCompletionOutput:
        return ChatCompletionOutput(
            model_info=self.model_info,
            message=self._parse_assistant_message(response['choices'][0]['message']),
            finish_reason=self._parse_finish_reason(response),
            usage=self._parse_usage(response),
            extra=self._parse_extra(response),
        )

    @override
    def _process_stream_response(
        self, response: dict[str, Any], stream_manager: StreamManager
    ) -> ChatCompletionStreamOutput | None:
        delta_dict = response['choices'][0].get('delta', {})
        self._update_delta(delta_dict, stream_manager=stream_manager)
        stream_manager.extra = self._parse_extra(response)
        stream_manager.usage = self._parse_usage(response)
        stream_manager.finish_reason = self._parse_finish_reason(response)
        return stream_manager.build_stream_output()

    def _parse_assistant_message(self, message: dict[str, Any]) -> AssistantMessage:
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
        return AssistantMessage(content=message.get('content') or '', tool_calls=tool_calls)

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
