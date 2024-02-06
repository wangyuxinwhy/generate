from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class DeepSeekSettings(BaseSettings):
    model_config = SettingsConfigDict(extra='ignore', env_prefix='deepseek_', env_file='.env')

    api_key: SecretStr
    api_base: str = 'https://api.deepseek.com/v1'
