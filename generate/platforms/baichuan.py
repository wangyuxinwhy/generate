from pydantic import SecretStr
from pydantic_settings import SettingsConfigDict

from generate.platforms.base import PlatformSettings


class BaichuanSettings(PlatformSettings):
    model_config = SettingsConfigDict(extra='ignore', env_prefix='baichuan_', env_file='.env')

    api_key: SecretStr
    secret_key: SecretStr
    api_base: str = 'https://api.baichuan-ai.com/v1'
    platform_url: str = 'https://platform.baichuan-ai.com/docs/api'
