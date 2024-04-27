from __future__ import annotations

from typing import Any, Protocol

from generate.chat_completion.message.core import (
    AssistantMessage,
    FunctionMessage,
    Messages,
    SystemMessage,
    ToolMessage,
    UserMessage,
    UserMultiPartMessage,
)


class MessageConverter(Protocol):
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
        raise NotImplementedError('FunctionMessage is not supported by this converter')

    def convert_tool_message(self, message: ToolMessage) -> dict[str, Any]:
        raise NotImplementedError('ToolMessage is not supported by this converter')

    def convert_user_multi_part_message(self, message: UserMultiPartMessage) -> dict[str, Any]:
        raise NotImplementedError('UserMultiPartMessage is not supported by this converter')
