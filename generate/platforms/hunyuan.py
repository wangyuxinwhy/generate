from pydantic import SecretStr
from pydantic_settings import SettingsConfigDict

from generate.platforms.base import PlatformSettings


class HunyuanSettings(PlatformSettings):
    model_config = SettingsConfigDict(extra='ignore', env_prefix='hunyuan_', env_file='.env')

    app_id: int
    secret_id: SecretStr
    secret_key: SecretStr
    completion_api: str = 'https://hunyuan.cloud.tencent.com/hyllm/v1/chat/completions'
    sign_api: str = 'hunyuan.cloud.tencent.com/hyllm/v1/chat/completions'
    platform_url: str = 'https://cloud.tencent.com/document/product/1729'
