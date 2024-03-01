from pydantic import SecretStr
from pydantic_settings import SettingsConfigDict

from generate.platforms.openai_like import OpenAILikeSettings


class YiSettings(OpenAILikeSettings):
    model_config = SettingsConfigDict(extra='ignore', env_prefix='yi_', env_file='.env')

    api_key: SecretStr
    api_base: str = 'https://api.lingyiwanwu.com/v1'
    platform_url: str = 'https://01ai.feishu.cn/docx/Q8Pcdn76uoHBc8xAvKCcPSd0nkc'
