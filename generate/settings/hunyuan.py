from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class HunyuanSettings(BaseSettings):
    model_config = SettingsConfigDict(extra='ignore', env_prefix='hunyuan_', env_file='.env')

    app_id: int
    secret_id: SecretStr
    secret_key: SecretStr
    completion_api: str = 'https://hunyuan.cloud.tencent.com/hyllm/v1/chat/completions'
    sign_api: str = 'hunyuan.cloud.tencent.com/hyllm/v1/chat/completions'
