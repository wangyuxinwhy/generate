from generate.platforms.azure import AzureSettings
from generate.platforms.baichuan import BaichuanSettings
from generate.platforms.baidu import BaiduCreationSettings, QianfanSettings
from generate.platforms.bailian import BailianSettings
from generate.platforms.dashscope import DashScopeSettings
from generate.platforms.hunyuan import HunyuanSettings
from generate.platforms.minimax import MinimaxSettings
from generate.platforms.moonshot import MoonshotSettings
from generate.platforms.openai import OpenAISettings
from generate.platforms.zhipu import ZhipuSettings

__all__ = [
    'AzureSettings',
    'BaichuanSettings',
    'BaiduCreationSettings',
    'MinimaxSettings',
    'ZhipuSettings',
    'OpenAISettings',
    'QianfanSettings',
    'BailianSettings',
    'HunyuanSettings',
    'DashScopeSettings',
    'MoonshotSettings',
]
