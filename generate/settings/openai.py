from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class OpenAISettings(BaseSettings):
    model_config = SettingsConfigDict(extra='ignore', env_prefix='openai_', env_file='.env')

    api_key: SecretStr
    api_base: str = 'https://api.openai.com/v1/'
