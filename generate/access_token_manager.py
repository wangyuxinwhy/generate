from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Optional


class AccessTokenManager(ABC):
    _token: Optional[str] = None
    _token_expires_at: datetime

    def __init__(self, token_refresh_days: int = 1) -> None:
        self._token = None
        self.token_refresh_days = token_refresh_days

    @property
    def token(self) -> str:
        if self._token is None:
            self._token = self._get_token()
            self._token_expires_at = datetime.now() + timedelta(days=self.token_refresh_days)
        else:
            self._maybe_refresh_token()
        return self._token

    @abstractmethod
    def _get_token(self) -> str:
        raise NotImplementedError

    def _maybe_refresh_token(self) -> None:
        if self._token_expires_at < datetime.now():
            self._token = self._get_token()
            self._token_expires_at = datetime.now() + timedelta(days=self.token_refresh_days)
