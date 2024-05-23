from __future__ import annotations

from typing import Any, Protocol, Type

from generate.chat_completion.message.core import (
    AssistantMessage,
    FunctionMessage,
    Message,
    Messages,
    SystemMessage,
    ToolMessage,
    UserMessage,
    UserMultiPartMessage,
)
from generate.chat_completion.message.exception import MessageTypeError


class MessageConverter(Protocol):
    allowed_message_types: list[Type[Message]]

    def convert_user_message(self, message: UserMessage) -> dict[str, Any]:
        ...

    def convert_assistant_message(self, message: AssistantMessage) -> dict[str, Any]:
        ...

    def convert_function_message(self, message: FunctionMessage) -> dict[str, Any]:
        ...

    def convert_tool_message(self, message: ToolMessage) -> dict[str, Any]:
        ...

    def convert_system_message(self, message: SystemMessage) -> dict[str, Any]:
        ...

    def convert_user_multi_part_message(self, message: UserMultiPartMessage) -> dict[str, Any]:
        ...

    def convert_message(self, message: Any) -> dict[str, Any]:
        convert_methods = {
            UserMessage: self.convert_user_message,
            AssistantMessage: self.convert_assistant_message,
            FunctionMessage: self.convert_function_message,
            ToolMessage: self.convert_tool_message,
            SystemMessage: self.convert_system_message,
            UserMultiPartMessage: self.convert_user_multi_part_message,
        }
        return convert_methods[type(message)](message)

    def convert_messages(self, messages: Messages) -> list[dict[str, Any]]:
        convert_methods = {
            UserMessage: self.convert_user_message,
            AssistantMessage: self.convert_assistant_message,
            FunctionMessage: self.convert_function_message,
            ToolMessage: self.convert_tool_message,
            SystemMessage: self.convert_system_message,
            UserMultiPartMessage: self.convert_user_multi_part_message,
        }
        return [convert_methods[type(message)](message) for message in messages]


class SimpleMessageConverter(MessageConverter):
    allowed_message_types = [UserMessage, AssistantMessage, SystemMessage]

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

    def convert_assistant_message(self, message: AssistantMessage) -> dict[str, Any]:
        return {
            'role': 'assistant',
            'content': message.content,
        }

    def convert_function_message(self, message: FunctionMessage) -> dict[str, Any]:
        raise MessageTypeError(message, allowed_message_type=list(self.allowed_message_types))

    def convert_tool_message(self, message: ToolMessage) -> dict[str, Any]:
        raise MessageTypeError(message, allowed_message_type=list(self.allowed_message_types))

    def convert_user_multi_part_message(self, message: UserMultiPartMessage) -> dict[str, Any]:
        raise MessageTypeError(message, allowed_message_type=list(self.allowed_message_types))
