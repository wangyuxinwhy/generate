from pydantic_settings import SettingsConfigDict

from generate.platforms.openai_like import OpenAILikeSettings


class BaichuanSettings(OpenAILikeSettings):
    model_config = SettingsConfigDict(extra='ignore', env_prefix='baichuan_', env_file='.env')

    api_base: str = 'https://api.baichuan-ai.com/v1'
    platform_url: str = 'https://platform.baichuan-ai.com/docs/api'
