from typing import Union

from typing_extensions import TypedDict

from generate.chat_completion.base import ModelParameters


class ToolChoice(TypedDict):
    name: str


class TestParameters(ModelParameters):
    name: str = 'TestModel'
    tool_choice: Union[str, ToolChoice, None] = None


def test_parameters() -> None:
    parameters = TestParameters(tool_choice=None)
    except_dump_data = {'name': 'TestModel', 'tool_choice': None}
    assert parameters.custom_model_dump() == except_dump_data
