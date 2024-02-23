from pydantic import SecretStr
from pydantic_settings import SettingsConfigDict

from generate.platforms.base import PlatformSettings


class MinimaxSettings(PlatformSettings):
    model_config = SettingsConfigDict(extra='ignore', env_prefix='minimax_', env_file='.env')

    group_id: str
    api_key: SecretStr
    api_base: str = 'https://api.minimax.chat/v1/'
    platform_url: str = 'https://api.minimax.chat/document/introduction'
