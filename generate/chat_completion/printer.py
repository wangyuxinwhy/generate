# ruff: noqa: T201
import time
from typing import Protocol

from generate.chat_completion.message import (
    AssistantMessage,
    FunctionMessage,
    Message,
    ToolMessage,
    UserMessage,
)
from generate.chat_completion.model_output import Stream


class MessagePrinter(Protocol):
    def print_message(self, message: Message) -> None:
        ...

    def print_stream(self, stream: Stream) -> None:
        ...


class SilentMessagePrinter(MessagePrinter):
    """
    A printer that does nothing.
    """

    def print_message(self, message: Message) -> None:
        pass

    def print_stream(self, stream: Stream) -> None:
        pass


class SimpleMessagePrinter(MessagePrinter):
    """
    A simple printer that prints messages and streams to the console.

    Args:
        smooth (bool, optional): Whether to use smooth printing. Defaults to True.
        interval (float, optional): The interval between each print. Defaults to 0.03.
    """

    def __init__(self, smooth: bool = True, interval: float = 0.03) -> None:
        self.smooth = smooth
        self.interval = interval

    def print_message(self, message: Message) -> None:
        if isinstance(message, (UserMessage, FunctionMessage, ToolMessage)):
            print(f'{message.role}: {message.content}')
        elif isinstance(message, AssistantMessage):
            if message.content:
                print(f'assistant: {message.content}')
            if message.function_call:
                print(f'Function call: {message.function_call.name}\nArguments: {message.function_call.arguments}')
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    print(
                        f'Tool call: {tool_call.id}\nFunction: {tool_call.function.name}\nArguments: {tool_call.function.arguments}'
                    )
        else:
            raise TypeError(f'Invalid message type: {type(message)}')

    def print_stream(self, stream: Stream) -> None:
        if stream.control == 'start':
            print('assistant: ', end='', flush=True)
        if self.smooth:
            for char in stream.delta:
                print(char, end='', flush=True)
                time.sleep(self.interval)
        else:
            print(stream.delta, end='', flush=True)
        if stream.control == 'finish':
            print()
