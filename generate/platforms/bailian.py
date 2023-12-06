from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from generate.access_token_manager import AccessTokenManager
from generate.http import HttpClient


class BailianSettings(BaseSettings):
    model_config = SettingsConfigDict(extra='ignore', env_prefix='bailian_', env_file='.env')

    default_app_id: str
    access_key_id: SecretStr
    access_key_secret: SecretStr
    agent_key: str
    completion_api: str = 'https://bailian.aliyuncs.com/v2/app/completions'


class BailianTokenManager(AccessTokenManager):
    def __init__(self, settings: BailianSettings, http_client: HttpClient, token_refresh_days: int = 1) -> None:
        super().__init__(token_refresh_days)
        self.settings = settings
        self.http_client = http_client

    def _get_token(self) -> str:
        try:
            import broadscope_bailian
        except ImportError as e:
            raise ImportError('Please install broadscope_bailian first: pip install broadscope_bailian') from e

        client = broadscope_bailian.AccessTokenClient(
            access_key_id=self.settings.access_key_id.get_secret_value(),
            access_key_secret=self.settings.access_key_secret.get_secret_value(),
            agent_key=self.settings.agent_key,
        )
        return client.get_token()
