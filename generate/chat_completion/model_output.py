from __future__ import annotations

from typing import Generic, Literal, Optional, TypeVar, cast

from pydantic import BaseModel

from generate.chat_completion.message import AssistantMessage, UnionAssistantMessage
from generate.model import ModelOutput

M = TypeVar('M', bound=UnionAssistantMessage)


class ChatCompletionOutput(ModelOutput, Generic[M]):
    message: M
    finish_reason: Optional[str] = None

    @property
    def reply(self) -> str:
        if self.message and isinstance(self.message, AssistantMessage):
            message = cast(AssistantMessage, self.message)
            return message.content
        return ''

    @property
    def is_finish(self) -> bool:
        return self.finish_reason is not None


class Stream(BaseModel):
    delta: str = ''
    control: Literal['start', 'continue', 'finish']


class ChatCompletionStreamOutput(ChatCompletionOutput, Generic[M]):
    stream: Stream
