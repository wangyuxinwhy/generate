from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class BailianSettings(BaseSettings):
    model_config = SettingsConfigDict(extra='ignore', env_prefix='bailian_', env_file='.env')

    app_id: str
    access_key_id: SecretStr
    access_key_secret: SecretStr
    agent_key: str
    completion_api: str = 'https://bailian.aliyuncs.com/v2/app/completions'
