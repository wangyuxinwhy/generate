from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class AzureSettings(BaseSettings):
    model_config = SettingsConfigDict(extra='ignore', env_prefix='azure_', env_file='.env')

    api_key: SecretStr
    api_base: str
    api_version: str
