from pydantic import SecretStr
from pydantic_settings import SettingsConfigDict

from generate.platforms.base import PlatformSettings


class AzureSettings(PlatformSettings):
    model_config = SettingsConfigDict(extra='ignore', env_prefix='azure_', env_file='.env')

    api_key: SecretStr
    api_base: str
    api_version: str
    platform_url: str = 'https://learn.microsoft.com/en-us/azure/ai-services/openai/'
