from datetime import datetime, timedelta
from typing import ClassVar, Optional


class TokenMixin:
    _token: Optional[str] = None
    _token_expires_at: datetime
    token_refresh_days: ClassVar[int] = 1

    @property
    def token(self) -> str:
        if self._token is None:
            self._token = self._get_token()
            self._token_expires_at = datetime.now() + timedelta(days=self.token_refresh_days)
        else:
            self.maybe_refresh_token()
        return self._token

    def _get_token(self) -> str:
        raise NotImplementedError

    def maybe_refresh_token(self) -> None:
        if self._token_expires_at < datetime.now():
            self._token = self._get_token()
            self._token_expires_at = datetime.now() + timedelta(days=self.token_refresh_days)
