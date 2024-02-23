from pydantic import SecretStr
from pydantic_settings import SettingsConfigDict

from generate.platforms.base import PlatformSettings


class DashScopeSettings(PlatformSettings):
    model_config = SettingsConfigDict(extra='ignore', env_prefix='dashscope_', env_file='.env')

    api_key: SecretStr
    api_base: str = 'https://dashscope.aliyuncs.com/api/v1'
    platform_url: str = 'https://help.aliyun.com/zh/dashscope/'
