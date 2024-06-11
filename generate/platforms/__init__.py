from generate.platforms.anthropic import AnthropicSettings
from generate.platforms.azure import AzureSettings
from generate.platforms.baichuan import BaichuanSettings
from generate.platforms.baidu import BaiduCreationSettings, QianfanSettings
from generate.platforms.base import PlatformSettings
from generate.platforms.dashscope import DashScopeSettings
from generate.platforms.deepseek import DeepSeekSettings
from generate.platforms.minimax import MinimaxSettings
from generate.platforms.moonshot import MoonshotSettings
from generate.platforms.openai import OpenAISettings
from generate.platforms.openrouter import OpenRouterSettings
from generate.platforms.stepfun import StepFunSettings
from generate.platforms.yi import YiSettings
from generate.platforms.zhipu import ZhipuSettings

__all__ = [
    'AzureSettings',
    'AnthropicSettings',
    'BaichuanSettings',
    'BaiduCreationSettings',
    'MinimaxSettings',
    'ZhipuSettings',
    'OpenAISettings',
    'QianfanSettings',
    'DashScopeSettings',
    'MoonshotSettings',
    'DeepSeekSettings',
    'YiSettings',
    'PlatformSettings',
    'StepFunSettings',
    'OpenRouterSettings',
]
