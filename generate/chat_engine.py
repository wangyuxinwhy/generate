from __future__ import annotations

import json
from typing import Any, Callable, List, Mapping, TypedDict

from typing_extensions import Self, Unpack

from generate.chat_completion import ChatCompletionModel, ChatCompletionOutput
from generate.chat_completion.message import (
    FunctionCall,
    FunctionMessage,
    ToolCall,
    ToolMessage,
    UnionMessage,
    UserMessage,
)
from generate.chat_completion.printer import MessagePrinter, SilentMessagePrinter, SimpleMessagePrinter
from generate.utils import load_chat_model


class ChatEngineKwargs(TypedDict, total=False):
    functions: Mapping[str, Callable] | None
    function_call_raise_error: bool
    max_calls_per_turn: int


class ChatEngine:
    """A chat engine for managing dialogues with a chat completion model.

    Args:
        chat_model (ChatCompletionModel): The chat completion model for generating responses.
        functions (Mapping[str, Callable] | None, optional): Functions to be used in the chat.
        function_call_raise_error (bool, optional): Whether to raise an error when calling a function fails. Defaults to False.
        max_calls_per_turn (int, optional): Maximum number of function calls allowed per turn. Defaults to 5.
        stream (bool | Literal['auto'], optional): Whether to use streaming. If 'auto', it is determined based on the presence of functions.
        printer (MessagePrinter | Literal['auto'] | None, optional): An instance for printing messages. If 'auto', a simple printer is used when streaming.
    """

    printer: MessagePrinter

    def __init__(
        self,
        chat_model: ChatCompletionModel,
        functions: Mapping[str, Callable] | None = None,
        function_call_raise_error: bool = False,
        max_calls_per_turn: int = 5,
        stream: bool = True,
        printer: MessagePrinter | None = SimpleMessagePrinter(),
    ) -> None:
        self._chat_model = chat_model
        self._function_map = functions or {}

        self.function_call_raise_error = function_call_raise_error
        self.max_calls_per_turn = max_calls_per_turn
        self.stream: bool = stream
        self.printer = printer or SilentMessagePrinter()
        self.history: list[UnionMessage] = []
        self.model_ouptuts: list[ChatCompletionOutput] = []
        self._call_count = 0

    @property
    def cost(self) -> float:
        return sum(output.cost or 0.0 for output in self.model_ouptuts)

    @classmethod
    def from_model_id(cls, model_id: str, **kwargs: Unpack[ChatEngineKwargs]) -> Self:
        chat_model = load_chat_model(model_id)
        return cls(chat_model, **kwargs)

    @property
    def chat_model(self) -> ChatCompletionModel:
        return self._chat_model

    def chat(self, user_input: str, **kwargs: Any) -> str:
        self._call_count = 0

        user_input_message = UserMessage(content=user_input)
        self.history.append(user_input_message)
        self.printer.print_message(user_input_message)

        while True:
            if self.stream:
                model_output = self._stream_chat_helper(**kwargs)
            else:
                model_output = self._chat_model.generate(self.history, **kwargs)
                self.printer.print_message(model_output.message)
            self._handle_model_output(model_output)
            if model_output.message.is_over:
                return model_output.reply

    def _handle_model_output(self, model_output: ChatCompletionOutput, **kwargs: Any) -> None:
        self.model_ouptuts.append(model_output)
        self.history.append(model_output.message)

        if model_output.message.function_call:
            self._call_count += 1
            if self._call_count > self.max_calls_per_turn:
                raise RuntimeError('Maximum number of function calls reached.')
            self._handle_function_call(model_output.message.function_call)

        if model_output.message.tool_calls:
            self._call_count += 1
            if self._call_count > self.max_calls_per_turn:
                raise RuntimeError('Maximum number of tool calls reached.')
            self._handle_tool_calls(model_output.message.tool_calls, **kwargs)

    def _handle_function_call(self, function_call: FunctionCall) -> None:
        function_output = self._run_function_call(function_call)
        function_message = FunctionMessage(name=function_call.name, content=json.dumps(function_output, ensure_ascii=False))
        self.history.append(function_message)
        if self.printer:
            self.printer.print_message(function_message)

    def _handle_tool_calls(self, tool_calls: List[ToolCall], **kwargs: Any) -> None:
        tool_messages: list[ToolMessage] = []
        for tool_call in tool_calls:
            funtion_output = self._run_function_call(tool_call.function)
            tool_message = ToolMessage(
                tool_call_id=tool_call.id, name=tool_call.function.name, content=json.dumps(funtion_output, ensure_ascii=False)
            )
            tool_messages.append(tool_message)
        self.history.extend(tool_messages)
        if self.printer:
            for message in tool_messages:
                self.printer.print_message(message)

    def _stream_chat_helper(self, **kwargs: Any) -> ChatCompletionOutput:
        for stream_output in self._chat_model.stream_generate(self.history, **kwargs):
            self.printer.print_stream(stream_output.stream)
            if stream_output.is_finish:
                return stream_output
        raise RuntimeError('Stream finished unexpectedly.')

    def _run_function_call(self, function_call: FunctionCall) -> Any | str:
        function = self._function_map.get(function_call.name)
        if function is None:
            if self.function_call_raise_error:
                raise ValueError(f'Function {function_call.name} not found')

            return 'Function not found, please try another function.'

        try:
            arguments = json.loads(function_call.arguments, strict=False)
            return function(**arguments)
        except Exception as e:
            if self.function_call_raise_error:
                raise

            return str(e)

    def reset(self) -> None:
        self.history.clear()
