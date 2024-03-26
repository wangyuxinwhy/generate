from pydantic import SecretStr
from pydantic_settings import SettingsConfigDict

from generate.platforms.openai_like import OpenAILikeSettings


class StepFunSettings(OpenAILikeSettings):
    model_config = SettingsConfigDict(extra='ignore', env_prefix='stepfun_', env_file='.env')

    api_key: SecretStr
    api_base: str = 'https://api.stepfun.com/v1'
    platform_url: str = 'https://platform.stepfun.com/'
