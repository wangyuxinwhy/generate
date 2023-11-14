from typing import Sequence, Type

from generate.chat_completion.message.core import Message


class MessageError(Exception):
    ...


class MessageTypeError(MessageError):
    def __init__(self, invalid_message: Message, allowed_message_type: Sequence[Type[Message]], *args: object) -> None:
        message = f'invalid message type: {type(invalid_message)}, only {tuple(allowed_message_type)} is allowed'
        super().__init__(message, *args)


class MessageValueError(MessageError):
    def __init__(self, invalid_message: Message, *args: object) -> None:
        message = f'invalid message value: {invalid_message}'
        super().__init__(message, *args)
