from __future__ import annotations

import json
from typing import Any, Callable, Generic, List, Literal, TypedDict, TypeVar, Union

from typing_extensions import Self, Unpack

from generate.chat_completion import ChatCompletionModel, ChatCompletionOutput, ModelParameters, function
from generate.chat_completion.message import (
    AssistantMessage,
    FunctionCall,
    FunctionCallMessage,
    FunctionMessage,
    Message,
    MessageTypeError,
    ToolCall,
    ToolCallsMessage,
    ToolMessage,
    UserMessage,
)
from generate.chat_completion.printer import MessagePrinter, SimpleMessagePrinter
from generate.utils import load_chat_model

P = TypeVar('P', bound=ModelParameters)


class ChatEngineKwargs(TypedDict, total=False):
    functions: Union[List[function], dict[str, Callable], None]
    call_raise_error: bool
    max_calls_per_turn: int


class ChatEngine(Generic[P]):
    """
    A chat engine that uses a chat model to generate responses.It will manage the chat history and function calls.

    Args:
        chat_model (BaseChatModel[P]): The chat model to use for generating responses.
        temperature (float | None, optional): Controls the randomness of the generated responses. Defaults to None.
        top_p (float | None, optional): Controls the diversity of the generated responses. Defaults to None.
        max_tokens (int | None, optional): Controls the length of the generated responses. Defaults to None.
        functions (Optional[List[function]], optional): A list of functions to use for function call. Defaults to None.
        function_call_raise_error (bool, optional): Whether to raise an error if a function call fails. Defaults to False.
        max_function_calls_per_turn (int, optional): The maximum number of function calls allowed per turn. Defaults to 5.
        stream (bool, optional): Whether to stream the generated text rely. Defaults to True.
        printer (Printer | Literal['auto'] | None, optional): The printer to use for displaying the generated responses. Defaults to 'auto'.
    """

    printer: MessagePrinter | None

    def __init__(
        self,
        chat_model: ChatCompletionModel[P],
        functions: Union[List[function], dict[str, Callable], None] = None,
        call_raise_error: bool = False,
        max_calls_per_turn: int = 5,
        stream: bool | Literal['auto'] = 'auto',
        printer: MessagePrinter | Literal['auto'] | None = 'auto',
    ) -> None:
        self._chat_model = chat_model

        if isinstance(functions, list):
            self._function_map = {}
            for i in functions:
                if not isinstance(i, function):
                    raise TypeError(f'Invalid function type: {type(i)}')
                self._function_map[i.json_schema['name']] = i
        elif isinstance(functions, dict):
            self._function_map = functions
        else:
            self._function_map = {}

        self.call_raise_error = call_raise_error
        self.max_calls_per_turn = max_calls_per_turn

        if stream == 'auto':
            self.stream = not bool(self._function_map)
        else:
            if self._function_map and stream:
                raise ValueError('Cannot stream when functions are provided.')
            self.stream = stream

        if printer == 'auto':
            self.printer = SimpleMessagePrinter() if stream else None
        else:
            self.printer = printer

        self.history: list[Message] = []
        self.model_ouptuts: list[ChatCompletionOutput] = []
        self._call_count = 0

    @classmethod
    def from_model_id(cls, model_id: str, **kwargs: Unpack[ChatEngineKwargs]) -> Self:
        chat_model = load_chat_model(model_id)
        return cls(chat_model, **kwargs)

    @property
    def chat_model(self) -> ChatCompletionModel[P]:
        return self._chat_model

    @property
    def print_message(self) -> bool:
        return self.printer is not None

    def chat(self, user_input: str, **override_parameters: Any) -> str:
        self._call_count = 0

        user_input_message = UserMessage(content=user_input)
        self.history.append(user_input_message)
        if self.printer:
            self.printer.print_message(user_input_message)

        while True:
            if self.stream:
                model_output = self._stream_chat_helper(**override_parameters)
            else:
                model_output = self._chat_model.generate(self.history, **override_parameters)
            self.handle_model_output(model_output)
            if isinstance(model_output.last_message, AssistantMessage):
                return model_output.reply

    def handle_model_output(self, model_output: ChatCompletionOutput, **override_parameters: Any) -> None:
        if not model_output.last_message:
            raise RuntimeError('messages in model output is empty.', model_output.model_dump())

        self.model_ouptuts.append(model_output)
        self.history.extend(model_output.messages)
        if self.printer:
            for message in model_output.messages:
                if self.stream and message.role == 'assistant':
                    continue
                self.printer.print_message(message)

        if isinstance(model_output.last_message, FunctionCallMessage):
            self._call_count += 1
            if self._call_count > self.max_calls_per_turn:
                raise RuntimeError('Maximum number of function calls reached.')
            function_call = model_output.last_message.content
            self._handle_function_call(function_call)

        if isinstance(model_output.last_message, ToolCallsMessage):
            self._call_count += 1
            if self._call_count > self.max_calls_per_turn:
                raise RuntimeError('Maximum number of tool calls reached.')
            tool_calls = model_output.last_message.content
            self._handle_tool_calls(tool_calls, **override_parameters)

    def _handle_function_call(self, function_call: FunctionCall) -> None:
        function_output = self.run_function_call(function_call)
        function_message = FunctionMessage(
            role='function', name=function_call.name, content=json.dumps(function_output, ensure_ascii=False)
        )
        self.history.append(function_message)
        if self.printer:
            self.printer.print_message(function_message)

    def _handle_tool_calls(self, tool_calls: List[ToolCall], **override_parameters: Any) -> None:
        tool_messages: list[ToolMessage] = []
        for tool_call in tool_calls:
            funtion_output = self.run_function_call(tool_call.function)
            tool_message = ToolMessage(
                tool_call_id=tool_call.id, name=tool_call.function.name, content=json.dumps(funtion_output, ensure_ascii=False)
            )
            tool_messages.append(tool_message)
        self.history.extend(tool_messages)
        if self.printer:
            for message in tool_messages:
                self.printer.print_message(message)

    def _stream_chat_helper(self, **override_parameters: Any) -> ChatCompletionOutput:
        for stream_output in self._chat_model.stream_generate(self.history, **override_parameters):
            if self.printer:
                self.printer.print_stream(stream_output.stream)
            if stream_output.is_finish:
                return stream_output
        raise RuntimeError('Stream finished unexpectedly.')

    async def async_chat(self, user_input: str, **override_parameters: Any) -> str | None:
        self._call_count = 0

        user_input_message = UserMessage(content=user_input)
        self.history.append(user_input_message)
        if self.printer:
            self.printer.print_message(user_input_message)

        while True:
            if self.stream:
                model_output = await self._async_stream_chat_helper(**override_parameters)
            else:
                model_output = await self._chat_model.async_generate(self.history, **override_parameters)
            self.handle_model_output(model_output)
            if isinstance(model_output.last_message, AssistantMessage):
                return model_output.reply

    async def _async_stream_chat_helper(self, **kwargs: Any) -> ChatCompletionOutput:
        async for stream_output in self._chat_model.async_stream_generate(self.history, **kwargs):
            if self.printer:
                self.printer.print_stream(stream_output.stream)
            if stream_output.is_finish:
                return stream_output
        raise RuntimeError('Stream finished unexpectedly.')

    def run_function_call(self, function_call: FunctionCall) -> Any | str:
        function = self._function_map.get(function_call.name)
        if function is None:
            if self.call_raise_error:
                raise ValueError(f'Function {function_call.name} not found')

            return 'Function not found, please try another function.'

        try:
            arguments = json.loads(function_call.arguments, strict=False)
            return function(**arguments)
        except Exception as e:
            if self.call_raise_error:
                raise

            return str(e)

    async def _async_recursive_function_call(self, function_call: FunctionCall, **override_parameters: Any) -> str:
        function_output = self.run_function_call(function_call)
        function_message = FunctionMessage(
            role='function', name=function_call.name, content=json.dumps(function_output, ensure_ascii=False)
        )
        self.history.append(function_message)
        if self.printer:
            self.printer.print_message(function_message)

        model_output = await self._chat_model.async_generate(self.history, **override_parameters)
        self.handle_model_output(model_output)

        if not model_output.last_message:
            raise RuntimeError('messages in model output is empty.', model_output.model_dump())

        if isinstance(model_output.last_message, AssistantMessage):
            return model_output.last_message.content

        if isinstance(model_output.last_message, FunctionCallMessage):
            self._call_count += 1
            if self._call_count > self.max_calls_per_turn:
                raise RuntimeError('Maximum number of function calls reached.')
            function_call = model_output.last_message.content
            return await self._async_recursive_function_call(function_call, **override_parameters)

        raise MessageTypeError(model_output.last_message, allowed_message_type=(AssistantMessage, FunctionCallMessage))

    def reset(self) -> None:
        self.history.clear()
