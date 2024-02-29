from pydantic import SecretStr
from pydantic_settings import SettingsConfigDict

from generate.platforms.base import PlatformSettings


class AnthropicSettings(PlatformSettings):
    model_config = SettingsConfigDict(extra='ignore', env_prefix='anthropic_', env_file='.env')

    api_key: SecretStr
    api_base: str = 'https://api.anthropic.com/v1'
    api_version: str = '2023-06-01'
    platform_url: str = 'https://docs.anthropic.com/'
