from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class BaichuanSettings(BaseSettings):
    model_config = SettingsConfigDict(extra='ignore', env_prefix='baichuan_', env_file='.env')

    api_key: SecretStr
    secret_key: SecretStr
    api_base: str = 'https://api.baichuan-ai.com/v1/chat'
    stream_api_base: str = 'https://api.baichuan-ai.com/v1/stream/chat'
