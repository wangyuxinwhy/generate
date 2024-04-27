from __future__ import annotations

from typing import Any, AsyncIterator, ClassVar, Dict, Iterator, List, Optional

from pydantic import Field, PositiveInt
from typing_extensions import Annotated, Unpack, override

from generate.chat_completion.base import RemoteChatCompletionModel
from generate.chat_completion.message import (
    AssistantMessage,
    Prompt,
)
from generate.chat_completion.message.converter import SimpleMessageConverter
from generate.chat_completion.message.core import FunctionCall, FunctionMessage, Messages, ToolCall, ToolMessage
from generate.chat_completion.model_output import ChatCompletionOutput, ChatCompletionStreamOutput, FinishReason, Usage
from generate.chat_completion.models.openai_like import OpenAITool, convert_to_openai_tool
from generate.chat_completion.stream_manager import StreamManager
from generate.chat_completion.tool import Tool, ToolCallMixin
from generate.http import (
    HttpClient,
    HttpxPostKwargs,
    ResponseValue,
)
from generate.model import ModelParameters, RemoteModelParametersDict
from generate.platforms.dashscope import DashScopeSettings
from generate.types import OrIterable, Probability
from generate.utils import ensure_iterable


class DashScopeChatParameters(ModelParameters):
    seed: Optional[PositiveInt] = None
    max_tokens: Optional[PositiveInt] = None
    top_p: Optional[Probability] = Field(default=None, alias='TopP')
    top_k: Optional[Annotated[int, Field(ge=0, le=100)]] = None
    repetition_penalty: Optional[float] = None
    temperature: Optional[Annotated[float, Field(gt=0, le=2)]] = None
    stop: Optional[List[str]] = None
    search: Annotated[Optional[bool], Field(alias='enable_search')] = None
    tools: Optional[List[OpenAITool]] = None


class DashScopeChatParametersDict(RemoteModelParametersDict, total=False):
    seed: Optional[PositiveInt]
    max_tokens: Optional[PositiveInt]
    top_p: Optional[Probability]
    top_k: Optional[Annotated[int, Field(ge=0, le=100)]]
    repetition_penalty: Optional[float]
    temperature: Optional[Annotated[float, Field(gt=0, le=2)]]
    stop: Optional[List[str]]
    search: Optional[bool]
    tools: Optional[List[OpenAITool]]


class DashScopeMessageConverter(SimpleMessageConverter):
    def convert_function_message(self, message: FunctionMessage) -> Dict[str, Any]:
        return {
            'role': 'tool',
            'name': message.name,
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
            base_dict['tool_calls'] = [
                {
                    'name': message.function_call.name,
                    'arguments': message.function_call.arguments,
                }
            ]
        return base_dict


class DashScopeChat(RemoteChatCompletionModel, ToolCallMixin):
    model_type: ClassVar[str] = 'dashscope'
    available_models: ClassVar[List[str]] = ['qwen-turbo', 'qwen-plus', 'qwen-max', 'qwen-max-longcontext']

    parameters: DashScopeChatParameters
    settings: DashScopeSettings
    message_converter: DashScopeMessageConverter

    def __init__(
        self,
        model: str = 'qwen-plus',
        parameters: DashScopeChatParameters | None = None,
        settings: DashScopeSettings | None = None,
        http_client: HttpClient | None = None,
        message_converter: DashScopeMessageConverter | None = None,
    ) -> None:
        parameters = parameters or DashScopeChatParameters()
        settings = settings or DashScopeSettings()  # type: ignore
        http_client = http_client or HttpClient()
        message_converter = message_converter or DashScopeMessageConverter()
        super().__init__(
            model=model, parameters=parameters, settings=settings, http_client=http_client, message_converter=message_converter
        )

    @override
    def generate(self, prompt: Prompt, **kwargs: Unpack[DashScopeChatParametersDict]) -> ChatCompletionOutput:
        return super().generate(prompt, **kwargs)

    @override
    async def async_generate(self, prompt: Prompt, **kwargs: Unpack[DashScopeChatParametersDict]) -> ChatCompletionOutput:
        return await super().async_generate(prompt, **kwargs)

    @override
    def stream_generate(
        self, prompt: Prompt, **kwargs: Unpack[DashScopeChatParametersDict]
    ) -> Iterator[ChatCompletionStreamOutput]:
        yield from super().stream_generate(prompt, **kwargs)

    @override
    async def async_stream_generate(
        self, prompt: Prompt, **kwargs: Unpack[DashScopeChatParametersDict]
    ) -> AsyncIterator[ChatCompletionStreamOutput]:
        async for output in super().async_stream_generate(prompt, **kwargs):
            yield output

    @override
    def _get_request_parameters(
        self, messages: Messages, stream: bool = False, **kwargs: Unpack[DashScopeChatParametersDict]
    ) -> HttpxPostKwargs:
        parameters = self.parameters.clone_with_changes(**kwargs)
        headers = {
            'Authorization': self.settings.api_key.get_secret_value(),
            'Content-Type': 'application/json',
        }
        if self.settings.workspace is not None:
            headers['X-DashScope-WorkSpace'] = self.settings.workspace
        if stream:
            headers['Accept'] = 'text/event-stream'

        params = {
            'input': {
                'messages': self.message_converter.convert_messages(messages),
            },
            'model': self.model,
            'parameters': parameters.custom_model_dump(),
        }
        return {
            'url': self.settings.api_base + '/services/aigc/text-generation/generation',
            'headers': headers,
            'json': params,
        }

    @override
    def _process_reponse(self, response: ResponseValue) -> ChatCompletionOutput:
        return ChatCompletionOutput(
            model_info=self.model_info,
            message=self._parse_assistant_message(response),
            usage=self._parse_usage(response),
            extra=self._parse_extra(response),
            finish_reason=self._parse_finish_reason(response['choices'][0]),
        )

    @override
    def _process_stream_response(
        self, response: dict[str, Any], stream_manager: StreamManager
    ) -> ChatCompletionStreamOutput | None:
        finish_reason = self._parse_finish_reason(response['output'])
        reply = response['output']['text']
        stream_manager.usage = self._parse_usage(response)
        stream_manager.extra = self._parse_extra(response)
        if finish_reason != 'null':
            stream_manager.finish_reason = finish_reason
            stream_manager.delta = ''
            return stream_manager.build_stream_output()
        stream_manager.delta = reply[len(stream_manager.content) :]
        return stream_manager.build_stream_output()

    @override
    def cost(self, input_tokens: int, output_tokens: int) -> Optional[float]:
        total_tokens = input_tokens + output_tokens
        model_price = {
            'qwen-turbo': 8,
            'qwen-plus': 20,
            'qwen-max': 120,
        }
        for model_name, price in model_price.items():
            if model_name in self.model:
                return total_tokens * price / 1_000_000
        return None

    @override
    def adapt_tool_calls(self, messages: Messages) -> None:
        tool_call_id_to_function_name = {}
        new_messages = []
        for message in messages:
            if isinstance(message, AssistantMessage) and message.tool_calls:
                for tool_call in message.tool_calls:
                    tool_call_id_to_function_name[tool_call.id] = tool_call.function.name
            if isinstance(message, ToolMessage):
                message = FunctionMessage(
                    name=tool_call_id_to_function_name[message.tool_call_id], content=message.content or ''
                )
            new_messages.append(message)
        messages[:] = new_messages

    @override
    def add_tools(self, tools: OrIterable[Tool]) -> None:
        new_tools = [convert_to_openai_tool(tool) for tool in ensure_iterable(tools)]
        if self.parameters.tools is None:
            self.parameters.tools = new_tools
        else:
            self.parameters.tools.extend(new_tools)

    def _parse_usage(self, response: dict[str, Any]) -> Usage:
        if usage := response.get('usage'):
            input_tokens = usage.get('input_tokens')
            output_tokens = usage.get('output_tokens')
            return Usage(input_tokens=input_tokens, output_tokens=output_tokens, cost=self.cost(input_tokens, output_tokens))
        return Usage()

    def _parse_assistant_message(self, response: dict[str, Any]) -> AssistantMessage:
        message = response['choices'][0]['message']
        if tool_calls_dict := message.get('tool_calls'):
            tool_calls = [
                ToolCall(
                    id=tool_call['function']['name'],
                    function=FunctionCall(
                        name=tool_call['function']['name'],
                        arguments=tool_call['function']['arguments'],
                    ),
                )
                for tool_call in tool_calls_dict
            ]
        else:
            tool_calls = None
        return AssistantMessage(content=message.get('content') or '', tool_calls=tool_calls)

    def _parse_extra(self, response: dict[str, Any]) -> Dict[str, Any]:
        return {
            'request_id': response['request_id'],
            'response': response,
        }

    def _parse_finish_reason(self, choice: dict[str, Any]) -> FinishReason | None:
        try:
            if finish_reason := choice.get('finish_reason'):
                return FinishReason(finish_reason)
        except (KeyError, IndexError, ValueError):
            return None
