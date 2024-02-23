from __future__ import annotations

import json
from typing import Any, AsyncGenerator, Generator, Sequence, Union

from typing_extensions import TypedDict

from generate.chat_completion import ChatCompletionModel
from generate.chat_completion.message.core import (
    AssistantMessage,
    FunctionCall,
    FunctionMessage,
    Messages,
    Prompt,
    ToolCall,
    ToolMessage,
)
from generate.chat_completion.message.utils import ensure_messages
from generate.chat_completion.model_output import ChatCompletionOutput
from generate.chat_completion.tool import Tool, ToolCallMixin, ToolDict
from generate.types import OrIterable

AgentMessage = Union[AssistantMessage, FunctionMessage, ToolMessage]


class AgentKwargs(TypedDict, total=False):
    tools: ToolDict | OrIterable[Tool] | None
    tool_call_raise_error: bool
    max_num_tool_calls_per_turn: int


class Agent(ChatCompletionModel):
    def __init__(
        self,
        model: ChatCompletionModel,
        tools: ToolDict | OrIterable[Tool] | None = None,
        tool_call_raise_error: bool = False,
        max_num_tool_calls_per_turn: int = 5,
    ) -> None:
        self.model = model

        if tools is None:
            self.tools = ToolDict()
        elif isinstance(tools, ToolDict):
            self.tools = tools
        else:
            self.tools = ToolDict.from_iterable(tools)

        if self.tools:
            if isinstance(model, ToolCallMixin):
                model.add_tools(self.tools.values())
            else:
                raise ValueError('Model does not support tools')

        self.tool_call_raise_error = tool_call_raise_error
        self.max_num_tool_calls_per_turn = max_num_tool_calls_per_turn

        self.history: Messages = []
        self.model_outputs: list[ChatCompletionOutput] = []
        self._call_count = 0

    def generate(self, prompt: Prompt, **kwargs: Any) -> ChatCompletionOutput:
        self._call_count = 0
        messages = ensure_messages(prompt)
        self.history.extend(messages)
        while True:
            model_output = self.model.generate(self.history, **kwargs)
            self.model_outputs.append(model_output)
            self.history.append(model_output.message)
            if model_output.message.is_over:
                return model_output
            self._handle_model_output(model_output)

    async def async_generate(self, prompt: Prompt, **kwargs: Any) -> ChatCompletionOutput:
        self._call_count = 0
        messages = ensure_messages(prompt)
        self.history.extend(messages)
        while True:
            model_output = await self.model.async_generate(self.history, **kwargs)
            self.model_outputs.append(model_output)
            self.history.append(model_output.message)
            if model_output.message.is_over:
                return model_output
            self._handle_model_output(model_output)

    def stream_generate(self, prompt: Prompt, **kwargs: Any) -> ChatCompletionOutput:
        raise NotImplementedError('Stream generate is not supported for agent, use stream_run')

    async def async_stream_generate(self, prompt: Prompt, **kwargs: Any) -> ChatCompletionOutput:
        raise NotImplementedError('Stream generate is not supported for agent, use async_stream_run')

    def run(self, prompt: Prompt, **kwargs: Any) -> Generator[AgentMessage, None, None]:
        self._call_count = 0
        messages = ensure_messages(prompt)
        self.history.extend(messages)

        while True:
            model_output = self.model.generate(self.history, **kwargs)
            yield model_output.message
            self.model_outputs.append(model_output)
            self.history.append(model_output.message)

            if model_output.message.is_over:
                return

            messages = self._handle_model_output(model_output)
            if messages:
                yield from messages

    def stream_run(self, prompt: Prompt, **kwargs: Any) -> Generator[AgentMessage, None, None]:
        self._call_count = 0
        messages = ensure_messages(prompt)
        self.history.extend(messages)

        while True:
            is_finish = False
            model_output = None
            finish_model_output = None
            for model_output in self.model.stream_generate(self.history, **kwargs):
                if not is_finish:
                    yield model_output.message
                if model_output.is_finish:
                    is_finish = True
                    finish_model_output = model_output
                    self.model_outputs.append(model_output)
                    self.history.append(model_output.message)

            assert finish_model_output is not None
            if finish_model_output.message.is_over or model_output is None:
                return

            messages = self._handle_model_output(model_output)
            if messages:
                yield from messages

    async def async_run(self, prompt: Prompt, **kwargs: Any) -> AsyncGenerator[AgentMessage, None]:
        self._call_count = 0
        messages = ensure_messages(prompt)
        self.history.extend(messages)

        while True:
            model_output = await self.model.async_generate(self.history, **kwargs)
            yield model_output.message
            self.model_outputs.append(model_output)
            self.history.append(model_output.message)

            if model_output.message.is_over:
                return

            messages = self._handle_model_output(model_output)
            if messages:
                for message in messages:
                    yield message

    async def async_stream_run(self, prompt: Prompt, **kwargs: Any) -> AsyncGenerator[AgentMessage, None]:
        self._call_count = 0
        messages = ensure_messages(prompt)
        self.history.extend(messages)

        while True:
            is_finish = False
            finish_model_output = None
            model_output = None
            async for model_output in self.model.async_stream_generate(self.history, **kwargs):
                if not is_finish:
                    yield model_output.message
                if model_output.is_finish:
                    is_finish = True
                    finish_model_output = model_output
                    self.model_outputs.append(model_output)
                    self.history.append(model_output.message)

            assert finish_model_output is not None
            if finish_model_output.message.is_over or model_output is None:
                return

            messages = self._handle_model_output(model_output)
            if messages:
                for message in messages:
                    yield message

    def _handle_model_output(self, model_output: ChatCompletionOutput) -> Sequence[AgentMessage] | None:
        is_function_call = model_output.message.function_call is not None
        is_tool_call = model_output.message.tool_calls is not None
        if is_function_call or is_tool_call:
            self._call_count += 1
            if self._call_count > self.max_num_tool_calls_per_turn:
                raise RuntimeError('Maximum number of function calls reached.')

        if model_output.message.function_call is not None:
            function_message = self._handle_function_call(model_output.message.function_call)
            self.history.append(function_message)
            return [function_message]

        if model_output.message.tool_calls is not None:
            tool_messages = self._handle_tool_calls(model_output.message.tool_calls)
            self.history.extend(tool_messages)
            return tool_messages
        return None

    def _handle_function_call(self, function_call: FunctionCall) -> FunctionMessage:
        if function_call.name not in self.tools:
            if self.tool_call_raise_error:
                raise ValueError(f'Function {function_call.name} not found')
            message_content = 'Function not found, please try another function.'
        else:
            try:
                arguments = json.loads(function_call.arguments, strict=False)
                function_output = self.tools.call(name=function_call.name, **arguments)
                message_content = json.dumps(function_output, ensure_ascii=False)
            except Exception as e:
                if self.tool_call_raise_error:
                    raise
                message_content = str(e)
        return FunctionMessage(name=function_call.name, content=message_content)

    def _handle_tool_calls(self, tool_calls: list[ToolCall]) -> list[ToolMessage]:
        tool_messages: list[ToolMessage] = []
        for tool_call in tool_calls:
            funtion_message = self._handle_function_call(tool_call.function)
            tool_message = ToolMessage(tool_call_id=tool_call.id, name=tool_call.function.name, content=funtion_message.content)
            tool_messages.append(tool_message)
        return tool_messages

    @property
    def name(self) -> str:
        return self.model.name

    @classmethod
    def from_name(cls, name: str) -> Agent:
        raise ValueError('Agent model cannot be created from name')

    def reset(self) -> None:
        self.history.clear()
