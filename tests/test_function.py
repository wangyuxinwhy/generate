from typing import Literal

from pydantic import BaseModel

from generate import ChatEngine, OpenAIChat, OpenAIChatParameters, function
from generate.chat_completion.function_call import get_json_schema


def get_weather_without_wrap(city: str, country: Literal['US', 'CN'] = 'US') -> str:
    """
    Returns a string describing the weather in a given city and country.

    Args:
        city (str): The name of the city.
        country (str, optional): The name of the country. Defaults to 'US'.

    Returns:
        str: A string describing the weather in the given city and country.
    """
    return f'{country}/{city}，晴朗，27度'


class UserInfo(BaseModel):
    name: str
    age: int


@function
def upload_user_info(user_info: UserInfo) -> Literal['success']:
    """
    Uploads user information to the database.

    Args:
        user_info (UserInfo): The user information to be uploaded.
    """
    return 'success'


def test_get_json_schema() -> None:
    json_schema = get_json_schema(get_weather_without_wrap)
    expected_json_schema = {
        'name': 'get_weather_without_wrap',
        'description': 'Returns a string describing the weather in a given city and country.',
        'parameters': {
            'properties': {
                'city': {'type': 'string', 'description': 'The name of the city.'},
                'country': {
                    'default': 'US',
                    'enum': ['US', 'CN'],
                    'type': 'string',
                    'description': "The name of the country. Defaults to 'US'.",
                },
            },
            'required': ['city'],
            'type': 'object',
        },
    }
    assert json_schema == expected_json_schema


def test_validate_function() -> None:
    output = upload_user_info(user_info={'name': 'John', 'age': 20})  # type: ignore
    assert output == 'success'


@function
def get_weather(loc: str) -> str:
    """
    获取指定地区的天气信息

    Parameters:
        loc: 地区，比如北京，上海等
    """
    return f'{loc}，晴朗，27度'


@function
def google(keyword: str) -> str:
    """
    搜索谷歌

    Parameters:
        keyword: 搜索关键词
    """
    return '没有内容'


def test_openai_function() -> None:
    model = OpenAIChat(parameters=OpenAIChatParameters(functions=[get_weather.json_schema, google.json_schema], temperature=0))
    engine = ChatEngine(
        model, functions={f.name: f for f in [get_weather, google]}, stream=False, function_call_raise_error=True
    )
    reply = engine.chat('今天北京天气怎么样？')
    assert '27' in reply


def test_openai_tool() -> None:
    model = OpenAIChat(
        parameters=OpenAIChatParameters(
            tools=[
                {'type': 'function', 'function': get_weather.json_schema},
                {'type': 'function', 'function': google.json_schema},
            ]
        )
    )
    engine = ChatEngine(
        model, functions={f.name: f for f in [get_weather, google]}, stream=False, function_call_raise_error=True
    )
    reply = engine.chat('今天北京天气怎么样？')
    assert '27' in reply
