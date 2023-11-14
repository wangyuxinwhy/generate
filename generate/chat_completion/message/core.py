from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Sequence, Union

from pydantic import BaseModel, Field, TypeAdapter
from typing_extensions import Annotated


class Message(BaseModel):
    role: str
    name: Optional[str] = None
    content: Any


class SystemMessage(Message):
    role: Literal['system'] = 'system'
    content: str


class UserMessage(Message):
    role: Literal['user'] = 'user'
    content_type: Literal['text'] = 'text'
    content: str


class TextPart(BaseModel):
    type: Literal['text'] = 'text'
    text: str


class ImageUrl(BaseModel):
    url: str
    detail: Optional[Literal['low', 'high', 'auto']] = None


class ImageUrlPart(BaseModel):
    type: Literal['image_url'] = 'image_url'
    image_url: ImageUrl


UserPartTypes = Annotated[Union[TextPart, ImageUrlPart], Field(discriminator='type')]


class UserMultiPartMessage(Message):
    role: Literal['user'] = 'user'
    content_type: Literal['multi_part'] = 'multi_part'
    content: List[UserPartTypes]


class FunctionMessage(Message):
    role: Literal['function'] = 'function'
    name: str
    content: str


class ToolMessage(Message):
    role: Literal['tool'] = 'tool'
    name: Optional[str] = None
    tool_call_id: str
    content: Optional[str] = None


class AssistantMessage(Message):
    role: Literal['assistant'] = 'assistant'
    content_type: Literal['text'] = 'text'
    content: str


class FunctionCall(BaseModel):
    name: str
    arguments: str
    thoughts: Optional[str] = None


class FunctionCallMessage(Message):
    role: Literal['assistant'] = 'assistant'
    content_type: Literal['function_call'] = 'function_call'
    content: FunctionCall


class ToolCall(BaseModel):
    id: str  # noqa: A003
    type: Literal['function'] = 'function'  # noqa: A003
    function: FunctionCall


class ToolCallsMessage(Message):
    role: Literal['assistant'] = 'assistant'
    content_type: Literal['tool_calls'] = 'tool_calls'
    content: List[ToolCall]


_AssistantMessage = Annotated[
    Union[AssistantMessage, FunctionCallMessage, ToolCallsMessage], Field(discriminator='content_type')
]
_UserMessage = Annotated[Union[UserMessage, UserMultiPartMessage], Field(discriminator='content_type')]
MessageTypes = Annotated[
    Union[SystemMessage, FunctionMessage, ToolMessage, _UserMessage, _AssistantMessage], Field(discriminator='role')
]
Messages = Sequence[Message]
MessageDict = Dict[str, Any]
MessageDicts = Sequence[MessageDict]
Prompt = Union[str, Message, Messages, MessageDict, MessageDicts]
Prompts = Sequence[Prompt]
AssistantContentTypes = Union[str, FunctionCall, List[ToolCall]]
UserContentTypes = Union[str, List[UserPartTypes]]
message_validator = TypeAdapter(MessageTypes)
assistant_content_validator = TypeAdapter(AssistantContentTypes)
user_content_validator = TypeAdapter(UserContentTypes)
