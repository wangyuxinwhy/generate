from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class MoonshotSettings(BaseSettings):
    model_config = SettingsConfigDict(extra='ignore', env_prefix='moonshot_', env_file='.env')

    api_key: SecretStr
    api_base: str = 'https://api.moonshot.cn/v1'
