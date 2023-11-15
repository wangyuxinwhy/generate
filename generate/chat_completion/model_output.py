from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from generate.chat_completion.message import AssistantMessage, Message, Messages
from generate.model import ModelOutput


class ChatCompletionOutput(ModelOutput):
    messages: Messages = []
    finish_reason: str = ''

    @property
    def last_message(self) -> Message | None:
        if self.messages:
            return self.messages[-1]
        return None

    @property
    def reply(self) -> str:
        if self.last_message and isinstance(self.last_message, AssistantMessage):
            return self.last_message.content
        return ''


class Stream(BaseModel):
    delta: str = ''
    control: Literal['start', 'continue', 'finish']


class ChatCompletionStreamOutput(ChatCompletionOutput):
    stream: Stream

    @property
    def is_finish(self) -> bool:
        return self.stream.control == 'finish'
