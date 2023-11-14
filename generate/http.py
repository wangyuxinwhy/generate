from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Union

from httpx._types import ProxiesTypes
from pydantic import BaseModel
from typing_extensions import Required, TypedDict

from generate.types import PrimitiveData

HttpResponse = Dict[str, Any]
QueryParams = Mapping[str, Union[PrimitiveData, Sequence[PrimitiveData]]]
Headers = Dict[str, str]


class RetryStrategy(BaseModel):
    min_wait_seconds: int = 2
    max_wait_seconds: int = 20
    max_attempt: int = 3


class HttpModelInitKwargs(TypedDict, total=False):
    timeout: Optional[int]
    retry: Union[bool, RetryStrategy]
    proxies: Union[ProxiesTypes, None]


class HttpxPostKwargs(TypedDict, total=False):
    url: Required[str]
    json: Required[Any]
    headers: Required[Headers]
    params: QueryParams
    timeout: Optional[int]


class UnexpectedResponseError(Exception):
    """
    Exception raised when an unexpected response is received from the server.

    Attributes:
        response (dict): The response from the server.
    """

    def __init__(self, response: dict, *args: Any) -> None:
        super().__init__(response, *args)
