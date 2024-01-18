import time

import cachetools.func  # type: ignore
import jwt
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

API_TOKEN_TTL_SECONDS = 3 * 60
CACHE_TTL_SECONDS = API_TOKEN_TTL_SECONDS - 30


class ZhipuSettings(BaseSettings):
    model_config = SettingsConfigDict(extra='ignore', env_prefix='zhipu_', env_file='.env')

    api_key: SecretStr
    v3_api_base: str = 'https://open.bigmodel.cn/api/paas/v3/model-api'
    v4_api_base: str = 'https://open.bigmodel.cn/api/paas/v4/'


@cachetools.func.ttl_cache(ttl=CACHE_TTL_SECONDS)
def generate_zhipu_token(api_key: str) -> str:
    try:
        api_key, secret = api_key.split('.')
    except Exception as e:
        raise ValueError('invalid api_key') from e

    payload = {
        'api_key': api_key,
        'exp': int(round(time.time() * 1000)) + API_TOKEN_TTL_SECONDS * 1000,
        'timestamp': int(round(time.time() * 1000)),
    }

    return jwt.encode(  # type: ignore
        payload,
        secret,
        algorithm='HS256',
        headers={'alg': 'HS256', 'sign_type': 'SIGN'},
    )
