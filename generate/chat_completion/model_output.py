from __future__ import annotations

from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel

from generate.chat_completion.message import AssistantMessage, Message, Messages
from generate.model import ModelOutput


class ChatCompletionModelOutput(ModelOutput):
    messages: Messages = []
    finish_reason: str = ''
    usage: Optional[Dict[str, int]] = None
    cost: Optional[float] = None

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
    control: Literal['start', 'continue', 'finish', 'done']


class FinishStream(Stream):
    control: Literal['finish'] = 'finish'
    finish_reason: str = ''
    usage: Dict[str, int] = {}
    cost: Optional[float] = None
    extra: Dict[str, Any] = {}


class ChatCompletionModelStreamOutput(ChatCompletionModelOutput):
    stream: Stream

    @property
    def is_finish(self) -> bool:
        return self.stream.control == 'finish'
