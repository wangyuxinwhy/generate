from pydantic import SecretStr
from pydantic_settings import SettingsConfigDict

from generate.platforms.base import PlatformSettings


class MoonshotSettings(PlatformSettings):
    model_config = SettingsConfigDict(extra='ignore', env_prefix='moonshot_', env_file='.env')

    api_key: SecretStr
    api_base: str = 'https://api.moonshot.cn/v1'
    platform_url: str = 'https://platform.moonshot.cn/docs'
