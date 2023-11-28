from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Sequence, Union

from pydantic import BaseModel, TypeAdapter


class Role(str, Enum):
    user = 'user'
    assistant = 'assistant'
    system = 'system'
    function = 'function'
    tool = 'tool'


class Message(BaseModel):
    role: Role
    name: Optional[str] = None
    content: Any


class SystemMessage(Message):
    role: Literal[Role.system] = Role.system
    content: str


class UserMessage(Message):
    role: Literal[Role.user] = Role.user
    content: str


class TextPart(BaseModel):
    text: str


class ImageUrl(BaseModel):
    url: str
    detail: Optional[Literal['low', 'high', 'auto']] = None


class ImageUrlPart(BaseModel):
    image_url: ImageUrl


class UserMultiPartMessage(Message):
    role: Literal[Role.user] = Role.user
    content: List[Union[TextPart, ImageUrlPart]]


class FunctionMessage(Message):
    role: Role = Role.function
    name: str
    content: str


class ToolMessage(Message):
    role: Literal[Role.tool] = Role.tool
    tool_call_id: str
    content: Optional[str] = None


class AssistantMessage(Message):
    role: Literal[Role.assistant] = Role.assistant
    content: str


class FunctionCall(BaseModel):
    name: str
    arguments: str
    thoughts: Optional[str] = None


class FunctionCallMessage(Message):
    role: Literal[Role.assistant] = Role.assistant
    content: FunctionCall


class ToolCall(BaseModel):
    id: str  # noqa: A003
    type: Literal['function'] = 'function'  # noqa: A003
    function: FunctionCall


class ToolCallsMessage(Message):
    role: Literal[Role.assistant] = Role.assistant
    name: Optional[str] = None
    content: List[ToolCall]


UnionAssistantMessage = Union[AssistantMessage, FunctionCallMessage, ToolCallsMessage]
UnionUserMessage = Union[UserMessage, UserMultiPartMessage]
UnionUserPart = Union[TextPart, ImageUrlPart]
UnionMessage = Union[SystemMessage, FunctionMessage, ToolMessage, UnionAssistantMessage, UnionUserMessage]
Messages = Sequence[UnionMessage]
MessageDict = Dict[str, Any]
MessageDicts = Sequence[MessageDict]
Prompt = Union[str, UnionMessage, Messages, MessageDict, MessageDicts]
Prompts = Sequence[Prompt]
AssistantContentTypes = Union[str, FunctionCall, List[ToolCall]]
UserContentTypes = Union[str, List[UnionUserPart]]
message_validator = TypeAdapter(UnionMessage)
assistant_content_validator = TypeAdapter(AssistantContentTypes)
user_content_validator = TypeAdapter(UserContentTypes)
