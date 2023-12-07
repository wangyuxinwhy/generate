from typing import Union

from typing_extensions import TypedDict

from generate.model import ModelParameters


class ToolChoice(TypedDict):
    name: str


class FakeParameters(ModelParameters):
    name: str = 'TestModel'
    tool_choice: Union[str, ToolChoice, None] = None


def test_parameters() -> None:
    parameters = FakeParameters(tool_choice=None)
    except_dump_data = {'name': 'TestModel', 'tool_choice': None}
    assert parameters.custom_model_dump() == except_dump_data
