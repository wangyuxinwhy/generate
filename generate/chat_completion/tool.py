from __future__ import annotations

import uuid
from collections import UserDict
from typing import Any, Callable, Generic, MutableMapping, Protocol, TypeVar, runtime_checkable

from docstring_parser import parse
from pydantic import TypeAdapter, validate_call
from typing_extensions import NotRequired, ParamSpec, Self, TypedDict

from generate.chat_completion.message.core import (
    AssistantMessage,
    FunctionCall,
    FunctionMessage,
    Messages,
    ToolCall,
    ToolMessage,
)
from generate.types import JsonSchema, OrIterable
from generate.utils import ensure_iterable

P = ParamSpec('P')
T = TypeVar('T')


class FunctionJsonSchema(TypedDict):
    name: str
    parameters: JsonSchema
    description: NotRequired[str]


def get_json_schema(function: Callable[..., Any]) -> FunctionJsonSchema:
    function_name = function.__name__
    docstring = parse(text=function.__doc__ or '')
    parameters = TypeAdapter(function).json_schema()
    for param in docstring.params:
        if (arg_name := param.arg_name) in parameters['properties'] and (description := param.description):
            parameters['properties'][arg_name]['description'] = description
    parameters['required'] = sorted(k for k, v in parameters['properties'].items() if 'default' not in v)
    recusive_remove(parameters, 'additionalProperties')
    recusive_remove(parameters, 'title')
    json_schema: FunctionJsonSchema = {
        'name': function_name,
        'description': docstring.short_description or '',
        'parameters': parameters,
    }
    return json_schema


def function(callable_obj: Callable[P, T]) -> Tool[P, T]:
    return Tool(callable_obj)


def tool(callable_obj: Callable[P, T]) -> Tool[P, T]:
    return Tool(callable_obj)


class Tool(Generic[P, T]):  # noqa: N801
    def __init__(self, callable_obj: Callable[P, T], name: str | None = None) -> None:
        self.callable_obj: Callable[P, T] = validate_call(callable_obj)
        self.json_schema: FunctionJsonSchema = get_json_schema(callable_obj)

        self._name = name

    @property
    def name(self) -> str:
        return self._name or self.json_schema['name']

    @property
    def description(self) -> str:
        return self.json_schema.get('description', '')

    @property
    def parameters(self) -> JsonSchema:
        return self.json_schema['parameters']

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
        return self.callable_obj(*args, **kwargs)


def recusive_remove(obj: Any, remove_key: str) -> None:
    """
    Recursively removes a key from a dictionary and all its nested dictionaries.

    Args:
        dictionary (dict): The dictionary to remove the key from.
        remove_key (str): The key to remove from the dictionary.

    Returns:
        None
    """
    if isinstance(obj, dict):
        for key in list(obj.keys()):
            if key == remove_key:
                del obj[key]
            else:
                recusive_remove(obj[key], remove_key)


class ToolDict(UserDict, MutableMapping[str, Tool]):
    def call(self, name: str, *args: Any, **kwargs: Any) -> Any:
        return self.data[name](*args, **kwargs)

    @classmethod
    def from_iterable(cls, tools: OrIterable[Tool]) -> Self:
        return cls({tool.name: tool for tool in ensure_iterable(tools)})


@runtime_checkable
class SupportToolCall(Protocol):
    def add_tools(self, tools: OrIterable[Tool]) -> None:
        ...

    def generate_tool_call_id(self, function_call: FunctionCall) -> str:
        return f'tool_{uuid.uuid4().hex}'

    def process_messages_for_tool_call(self, messages: Messages) -> None:
        for index in range(len(messages)):
            current_message = messages[index]
            if isinstance(current_message, AssistantMessage) and current_message.function_call is not None:
                tool_call_id = self.generate_tool_call_id(current_message.function_call)
                messages[index].tool_calls = [ToolCall(id=tool_call_id, function=current_message.function_call)]
                messages[index].function_call = None
                next_message = messages[index + 1] if index + 1 < len(messages) else None
                if next_message is not None and isinstance(next_message, FunctionMessage):
                    messages[index + 1] = ToolMessage(
                        tool_call_id=tool_call_id, name=next_message.name, content=next_message.content
                    )
