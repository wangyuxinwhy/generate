from __future__ import annotations

from typing import Generic, List, Literal, Optional, TypeVar

from pydantic import BaseModel

from generate.chat_completion.message import AssistantMessage, UnionMessage
from generate.model import ModelOutput

M = TypeVar('M', bound=UnionMessage)


class ChatCompletionOutput(ModelOutput, Generic[M]):
    messages: List[M] = []
    finish_reason: Optional[str] = None

    @property
    def message(self) -> M:
        if len(self.messages) != 1:
            raise ValueError('Expected exactly one message')
        return self.messages[0]

    @property
    def reply(self) -> str:
        if self.message and isinstance(self.message, AssistantMessage):
            return self.message.content
        return ''

    @property
    def is_finish(self) -> bool:
        return self.finish_reason is not None


class Stream(BaseModel):
    delta: str = ''
    control: Literal['start', 'continue', 'finish']


class ChatCompletionStreamOutput(ChatCompletionOutput, Generic[M]):
    stream: Stream
