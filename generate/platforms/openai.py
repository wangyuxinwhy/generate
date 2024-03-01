from pydantic import SecretStr
from pydantic_settings import SettingsConfigDict

from generate.platforms.openai_like import OpenAILikeSettings


class OpenAISettings(OpenAILikeSettings):
    model_config = SettingsConfigDict(extra='ignore', env_prefix='openai_', env_file='.env')

    api_key: SecretStr
    api_base: str = 'https://api.openai.com/v1'
    platform_url: str = 'https://platform.openai.com/docs'
