from __future__ import annotations

import json
from typing import Callable, Generic, TypeVar

from docstring_parser import parse
from pydantic import TypeAdapter, validate_call
from typing_extensions import NotRequired, ParamSpec, TypedDict

from generate.chat_completion.message import FunctionCallMessage, Message
from generate.types import JsonSchema

P = ParamSpec('P')
T = TypeVar('T')


class FunctionJsonSchema(TypedDict):
    name: str
    parameters: JsonSchema
    description: NotRequired[str]


def get_json_schema(function: Callable) -> FunctionJsonSchema:
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


class function(Generic[P, T]):  # noqa: N801
    """
    A decorator class that wraps a callable function and provides additional functionality.

    Args:
        function (Callable[P, T]): The function to be wrapped.

    Attributes:
        function (Callable[P, T]): The wrapped function.
        name (str): The name of the wrapped function.
        docstring (ParsedDocstring): The parsed docstring of the wrapped function.
        json_schema (Function): The JSON schema of the wrapped function.

    Methods:
        __call__(self, *args: Any, **kwargs: Any) -> Any: Calls the wrapped function with the provided arguments.
        call_with_message(self, message: Message) -> T: Calls the wrapped function with the arguments provided in the message.
    """

    def __init__(self, function: Callable[P, T]) -> None:
        self.function: Callable[P, T] = validate_call(function)
        self.json_schema = get_json_schema(function)

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
        return self.function(*args, **kwargs)

    def call_with_message(self, message: Message) -> T:
        if isinstance(message, FunctionCallMessage):
            function_call = message.content
            arguments = json.loads(function_call.arguments, strict=False)
            return self.function(**arguments)  # type: ignore
        raise ValueError(f'message is not a function call: {message}')


def recusive_remove(dictionary: dict, remove_key: str) -> None:
    """
    Recursively removes a key from a dictionary and all its nested dictionaries.

    Args:
        dictionary (dict): The dictionary to remove the key from.
        remove_key (str): The key to remove from the dictionary.

    Returns:
        None
    """
    if isinstance(dictionary, dict):
        for key in list(dictionary.keys()):
            if key == remove_key:
                del dictionary[key]
            else:
                recusive_remove(dictionary[key], remove_key)
