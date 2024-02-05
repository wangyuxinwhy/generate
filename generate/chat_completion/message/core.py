from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Sequence, Union

from pydantic import BaseModel, TypeAdapter


class Message(BaseModel):
    role: str
    name: Optional[str] = None
    content: Any


class SystemMessage(Message):
    role: Literal['system'] = 'system'
    content: str


class UserMessage(Message):
    role: Literal['user'] = 'user'
    content: str


class TextPart(BaseModel):
    text: str


class ImageUrl(BaseModel):
    url: str
    detail: Optional[Literal['low', 'high', 'auto']] = None


class ImageUrlPart(BaseModel):
    image_url: ImageUrl


class ImagePart(BaseModel):
    image: bytes
    image_format: Optional[str] = None


class UserMultiPartMessage(Message):
    role: Literal['user'] = 'user'
    content: List[Union[TextPart, ImageUrlPart, ImagePart]]


class FunctionMessage(Message):
    role: Literal['function'] = 'function'
    name: str
    content: str


class ToolMessage(Message):
    role: Literal['tool'] = 'tool'
    tool_call_id: str
    content: Optional[str] = None


class FunctionCall(BaseModel):
    name: str
    arguments: str
    thoughts: Optional[str] = None


class ToolCall(BaseModel):
    id: str  # noqa: A003
    type: str = 'function'
    function: FunctionCall


class AssistantMessage(Message):
    role: Literal['assistant'] = 'assistant'
    content: str = ''
    function_call: Optional[FunctionCall] = None
    tool_calls: Optional[List[ToolCall]] = None

    @property
    def is_over(self) -> bool:
        return self.function_call is None and self.tool_calls is None


UnionUserMessage = Union[UserMessage, UserMultiPartMessage]
UnionUserPart = Union[TextPart, ImageUrlPart]
UnionMessage = Union[SystemMessage, FunctionMessage, ToolMessage, AssistantMessage, UnionUserMessage]
Messages = List[UnionMessage]
MessageDict = Dict[str, Any]
MessageDicts = Sequence[MessageDict]
Prompt = Union[str, UnionMessage, Messages, MessageDict, MessageDicts]
Prompts = Sequence[Prompt]
AssistantContentTypes = Union[str, FunctionCall, List[ToolCall]]
UserContentTypes = Union[str, List[UnionUserPart]]
message_validator = TypeAdapter(UnionMessage)
assistant_content_validator = TypeAdapter(AssistantContentTypes)
user_content_validator = TypeAdapter(UserContentTypes)
