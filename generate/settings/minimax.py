from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class MinimaxSettings(BaseSettings):
    model_config = SettingsConfigDict(extra='ignore', env_prefix='minimax_', env_file='.env')

    group_id: str
    api_key: SecretStr
    api_base: str = 'https://api.minimax.chat/v1/'
