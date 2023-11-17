from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class ZhipuSettings(BaseSettings):
    model_config = SettingsConfigDict(extra='ignore', env_prefix='zhipu_', env_file='.env')

    api_key: SecretStr
    api_base: str = 'https://open.bigmodel.cn/api/paas/v3/model-api'
