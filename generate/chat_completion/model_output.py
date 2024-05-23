from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel

from generate.chat_completion.message import AssistantMessage
from generate.model import ModelOutput


class FinishReason(str, Enum):
    end_turn = 'end_turn'
    stop = 'stop'
    length = 'length'
    content_filter = 'content_filter'
    tool_calls = 'tool_calls'
    funtion_call = 'function_call'


class Usage(BaseModel):
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None


class ChatCompletionOutput(ModelOutput):
    message: AssistantMessage
    usage: Optional[Usage] = None
    finish_reason: Optional[FinishReason] = None

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
