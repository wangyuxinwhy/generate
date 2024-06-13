from typing import Literal

from pydantic import BaseModel

from generate import OpenAIChat, tool, MoonshotChat
from generate.chat_completion.tool import get_json_schema
from dotenv import load_dotenv

load_dotenv()  # 加载.env文件

@tool
def get_weather(loc: str) -> str:
    """
    获取指定地区的天气信息

    Parameters:
        loc: 地区，比如北京，上海等
    """
    return f'{loc}，晴朗，27度'

def test_moonshot() -> None:
    moonshot = MoonshotChat()
    resp = moonshot.generate('今天北京天气怎么样？', tools=[{"type":"function", "function": get_weather.json_schema}], tool_choice='auto')
    assert resp.message.tool_calls[0].function.name == "get_weather" # type: ignore
    
