from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class WenxinSettings(BaseSettings):
    model_config = SettingsConfigDict(extra='ignore', env_prefix='wenxin_', env_file='.env')

    api_key: SecretStr
    secret_key: SecretStr
    comlpetion_api_base: str = 'https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/'
    access_token_api: str = 'https://aip.baidubce.com/oauth/2.0/token'
