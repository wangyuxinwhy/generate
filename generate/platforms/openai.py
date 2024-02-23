from pydantic import SecretStr
from pydantic_settings import SettingsConfigDict

from generate.platforms.base import PlatformSettings


class OpenAISettings(PlatformSettings):
    model_config = SettingsConfigDict(extra='ignore', env_prefix='openai_', env_file='.env')

    api_key: SecretStr
    api_base: str = 'https://api.openai.com/v1/'
    platform_url: str = 'https://platform.openai.com/docs'
