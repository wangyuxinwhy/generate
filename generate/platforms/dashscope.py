from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class DashScopeSettings(BaseSettings):
    model_config = SettingsConfigDict(extra='ignore', env_prefix='dashscope_', env_file='.env')

    api_key: SecretStr
    api_base: str = 'https://dashscope.aliyuncs.com/api/v1'
