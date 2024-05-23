from typing import Optional

from pydantic_settings import SettingsConfigDict

from generate.platforms.openai_like import OpenAILikeSettings


class AzureSettings(OpenAILikeSettings):
    model_config = SettingsConfigDict(extra='ignore', env_prefix='azure_', env_file='.env')

    api_version: str
    chat_api_engine: Optional[str] = None
    platform_url: str = 'https://learn.microsoft.com/en-us/azure/ai-services/openai/'
