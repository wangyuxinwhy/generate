from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel

from generate.chat_completion.message import AssistantMessage
from generate.model import ModelOutput


class ChatCompletionOutput(ModelOutput):
    message: AssistantMessage
    finish_reason: Optional[str] = None

    @property
    def reply(self) -> str:
        return self.message.content

    @property
    def is_finish(self) -> bool:
        return self.finish_reason is not None


class Stream(BaseModel):
    delta: str = ''
    control: Literal['start', 'continue', 'finish']


class ChatCompletionStreamOutput(ChatCompletionOutput):
    stream: Stream
