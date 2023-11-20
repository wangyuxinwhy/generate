from __future__ import annotations

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from generate.access_token_manager import AccessTokenManager
from generate.http import HttpClient, UnexpectedResponseError


class QianfanSettings(BaseSettings):
    model_config = SettingsConfigDict(extra='ignore', env_prefix='qianfan_', env_file='.env')

    api_key: SecretStr
    secret_key: SecretStr
    comlpetion_api_base: str = 'https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/'
    image_generation_api_base: str = 'https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/text2image/'
    access_token_api: str = 'https://aip.baidubce.com/oauth/2.0/token'


class QianfanTokenManager(AccessTokenManager):
    def __init__(self, settings: QianfanSettings, http_client: HttpClient, token_refresh_days: int = 20) -> None:
        super().__init__(token_refresh_days)
        self.settings = settings
        self.http_client = http_client

    def _get_token(self) -> str:
        headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
        params = {
            'grant_type': 'client_credentials',
            'client_id': self.settings.api_key.get_secret_value(),
            'client_secret': self.settings.secret_key.get_secret_value(),
        }
        response = self.http_client.post(
            {
                'url': self.settings.access_token_api,
                'headers': headers,
                'params': params,
                'json': None,
            }
        )
        response_dict = response.json()
        if 'error' in response_dict:
            raise UnexpectedResponseError(response_dict)
        return response_dict['access_token']


class BaiduCreationSettings(BaseSettings):
    model_config = SettingsConfigDict(extra='ignore', env_prefix='baidu_creation_', env_file='.env')

    api_key: SecretStr
    secret_key: SecretStr
    image_generation_api: str = 'https://aip.baidubce.com/rpc/2.0/ernievilg/v1/txt2imgv2'
    access_token_api: str = 'https://aip.baidubce.com/oauth/2.0/token'


class BaiduCreationTokenManager(AccessTokenManager):
    def __init__(self, settings: BaiduCreationSettings, http_client: HttpClient, token_refresh_days: int = 5) -> None:
        super().__init__(token_refresh_days)
        self.settings = settings
        self.http_client = http_client

    def _get_token(self) -> str:
        headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
        params = {
            'grant_type': 'client_credentials',
            'client_id': self.settings.api_key.get_secret_value(),
            'client_secret': self.settings.secret_key.get_secret_value(),
        }
        response = self.http_client.post(
            {
                'url': self.settings.access_token_api,
                'headers': headers,
                'params': params,
                'json': None,
            }
        )
        response_dict = response.json()
        if 'error' in response_dict:
            raise UnexpectedResponseError(response_dict)
        return response_dict['access_token']
