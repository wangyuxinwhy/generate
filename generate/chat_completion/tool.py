from __future__ import annotations

from collections import UserDict
from typing import Any, Callable, Generic, MutableMapping, TypeVar

from docstring_parser import parse
from pydantic import TypeAdapter, validate_call
from typing_extensions import NotRequired, ParamSpec, Self, TypedDict

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


class ToolCallMixin:
    def add_tools(self, tools: OrIterable[Tool]) -> None:
        raise NotImplementedError
