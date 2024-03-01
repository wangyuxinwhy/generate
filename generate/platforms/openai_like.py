from abc import ABC

from pydantic import SecretStr

from generate.platforms.base import PlatformSettings


class OpenAILikeSettings(PlatformSettings, ABC):
    api_key: SecretStr
    api_base: str
