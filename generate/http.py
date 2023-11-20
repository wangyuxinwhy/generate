from __future__ import annotations

import logging
from typing import Any, AsyncGenerator, Dict, Generator, Literal, Mapping, Optional, Sequence, Union, overload

import httpx
from httpx import Limits, Response
from httpx._types import ProxiesTypes
from httpx_sse import aconnect_sse, connect_sse
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_random_exponential
from typing_extensions import Required, TypedDict

from generate.types import PrimitiveData

logger = logging.getLogger(__name__)
ResponseValue = Dict[str, Any]
QueryParams = Mapping[str, Union[PrimitiveData, Sequence[PrimitiveData]]]
Headers = Dict[str, str]


class RetryStrategy(BaseModel):
    min_wait_seconds: int = 2
    max_wait_seconds: int = 20
    max_attempt: int = 3


class HttpGetKwargs(TypedDict, total=False):
    url: Required[str]
    params: QueryParams
    headers: Headers
    timeout: Optional[int]


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
        response (dict[str, Any]): The response from the server.
    """

    def __init__(self, response: ResponseValue, *args: Any) -> None:
        super().__init__(response, *args)


class HttpClient:
    """
    A class representing an HTTP client.

    Args:
        retry (bool | RetryStrategy, optional): Whether to enable retry for failed requests. Defaults to False.
        timeout (int | None, optional): The timeout value for requests in seconds. Defaults to 60.
        proxies (ProxiesTypes | None, optional): The proxies to be used for requests. Defaults to None.
        stream_strategy (Literal['sse', 'basic'], optional): The strategy for streaming requests. Defaults to 'sse'.
        limits (Limits | None, optional): The limits for the HTTP client. Defaults to None.
    """

    def __init__(
        self,
        retry: bool | RetryStrategy = False,
        timeout: int | None = 60,
        proxies: ProxiesTypes | None = None,
        stream_strategy: Literal['sse', 'basic'] = 'sse',
        limits: Limits | None = None,
    ) -> None:
        self._timeout = timeout
        self._limits = limits or Limits(max_connections=100, max_keepalive_connections=20, keepalive_expiry=600)
        if isinstance(retry, RetryStrategy):
            self._retry = retry
        else:
            self._retry = RetryStrategy() if retry else None
        self._proxies = proxies
        self.stream_strategy = stream_strategy
        self.client = self.create_httpx_client(is_async=False)
        self.async_client = self.create_httpx_client(is_async=True)

    @property
    def timeout(self) -> int | None:
        return self._timeout

    @timeout.setter
    def timeout(self, timeout: int) -> None:
        self._timeout = timeout
        self.client = self.create_httpx_client(is_async=False)
        self.async_client = self.create_httpx_client(is_async=True)

    @property
    def retry(self) -> RetryStrategy | None:
        return self._retry

    @retry.setter
    def retry(self, retry: bool | RetryStrategy) -> None:
        if isinstance(retry, RetryStrategy):
            self._retry = retry
        else:
            self._retry = RetryStrategy() if retry else None
        self.client = self.create_httpx_client(is_async=False)
        self.async_client = self.create_httpx_client(is_async=True)

    @property
    def proxies(self) -> ProxiesTypes | None:
        return self._proxies

    @proxies.setter
    def proxies(self, proxies: ProxiesTypes | None) -> None:
        self._proxies = proxies
        self.client = self.create_httpx_client(is_async=False)
        self.async_client = self.create_httpx_client(is_async=True)

    @property
    def limits(self) -> Limits:
        return self._limits

    @limits.setter
    def limits(self, limits: Limits) -> None:
        self._limits = limits
        self.client = self.create_httpx_client(is_async=False)
        self.async_client = self.create_httpx_client(is_async=True)

    @overload
    def create_httpx_client(self, is_async: Literal[False]) -> httpx.Client:
        ...

    @overload
    def create_httpx_client(self, is_async: Literal[True]) -> httpx.AsyncClient:
        ...

    def create_httpx_client(self, is_async: bool) -> httpx.AsyncClient | httpx.Client:
        if is_async:
            return httpx.AsyncClient(proxies=self._proxies, timeout=self._timeout, limits=self._limits)
        return httpx.Client(proxies=self._proxies, timeout=self._timeout, limits=self._limits)

    def get(self, request_parameters: HttpGetKwargs) -> Response:
        if self.retry is None:
            return self._get(request_parameters)

        wait = wait_random_exponential(min=self.retry.min_wait_seconds, max=self.retry.max_wait_seconds)
        stop = stop_after_attempt(self.retry.max_attempt)
        return retry(wait=wait, stop=stop)(self._get)(request_parameters)

    async def async_get(self, request_parameters: HttpGetKwargs) -> Response:
        if self.retry is None:
            return await self._async_get(request_parameters)

        wait = wait_random_exponential(min=self.retry.min_wait_seconds, max=self.retry.max_wait_seconds)
        stop = stop_after_attempt(self.retry.max_attempt)
        return await retry(wait=wait, stop=stop)(self._async_get)(request_parameters)

    def post(self, request_parameters: HttpxPostKwargs) -> Response:
        if self.retry is None:
            return self._post(request_parameters)

        wait = wait_random_exponential(min=self.retry.min_wait_seconds, max=self.retry.max_wait_seconds)
        stop = stop_after_attempt(self.retry.max_attempt)
        return retry(wait=wait, stop=stop)(self._post)(request_parameters)

    async def async_post(self, request_parameters: HttpxPostKwargs) -> Response:
        if self.retry is None:
            return await self._async_post(request_parameters)

        wait = wait_random_exponential(min=self.retry.min_wait_seconds, max=self.retry.max_wait_seconds)
        stop = stop_after_attempt(self.retry.max_attempt)
        return await retry(wait=wait, stop=stop)(self._async_post)(request_parameters)

    def stream_post(self, request_parameters: HttpxPostKwargs) -> Generator[str, None, None]:
        if self.retry is None:
            return self._stream_post(request_parameters)

        wait = wait_random_exponential(min=self.retry.min_wait_seconds, max=self.retry.max_wait_seconds)
        stop = stop_after_attempt(self.retry.max_attempt)
        return retry(wait=wait, stop=stop)(self._stream_post)(request_parameters)

    def async_stream_post(self, request_parameters: HttpxPostKwargs) -> AsyncGenerator[str, None]:
        if self.retry is None:
            return self._async_stream_post(request_parameters)

        wait = wait_random_exponential(min=self.retry.min_wait_seconds, max=self.retry.max_wait_seconds)
        stop = stop_after_attempt(self.retry.max_attempt)
        return retry(wait=wait, stop=stop)(self._async_stream_post)(request_parameters)

    def _post(self, request_parameters: HttpxPostKwargs) -> Response:
        logger.debug(f'POST {request_parameters}')
        http_response = self.client.post(**request_parameters)  # type: ignore
        http_response.raise_for_status()
        logger.debug(f'Response {http_response}')
        return http_response

    async def _async_post(self, request_parameters: HttpxPostKwargs) -> Response:
        logger.debug(f'POST {request_parameters}')
        http_response = await self.async_client.post(**request_parameters)  # type: ignore
        http_response.raise_for_status()
        logger.debug(f'Response {http_response}')
        return http_response

    def _get(self, request_parameters: HttpGetKwargs) -> Response:
        logger.debug(f'GET {request_parameters}')
        http_response = self.client.get(**request_parameters)
        http_response.raise_for_status()
        return http_response

    async def _async_get(self, request_parameters: HttpGetKwargs) -> Response:
        logger.debug(f'GET {request_parameters}')
        http_response = await self.async_client.get(**request_parameters)
        http_response.raise_for_status()
        return http_response

    def _stream_post(self, request_parameters: HttpxPostKwargs) -> Generator[str, None, None]:
        logger.debug(f'POST {request_parameters}')
        if self.stream_strategy == 'sse':
            return self._generate_data_from_sse_stream(request_parameters)
        return self._generate_data_from_basic_stream(request_parameters)

    def _generate_data_from_sse_stream(self, request_parameters: HttpxPostKwargs) -> Generator[str, None, None]:
        with connect_sse(client=self.client, method='POST', **request_parameters) as source:
            for sse in source.iter_sse():
                yield sse.data

    def _generate_data_from_basic_stream(self, request_parameters: HttpxPostKwargs) -> Generator[str, None, None]:
        with self.client.stream('POST', **request_parameters) as source:
            for line in source.iter_lines():
                yield line

    def _async_stream_post(self, request_parameters: HttpxPostKwargs) -> AsyncGenerator[str, None]:
        logger.debug(f'POST {request_parameters}')
        if self.stream_strategy == 'sse':
            return self._async_generate_data_from_sse_stream(request_parameters)
        return self._async_generate_data_from_basic_stream(request_parameters)

    async def _async_generate_data_from_sse_stream(self, http_parameters: HttpxPostKwargs) -> AsyncGenerator[str, None]:
        async with aconnect_sse(client=self.async_client, method='POST', **http_parameters) as source:
            async for sse in source.aiter_sse():
                yield sse.data

    async def _async_generate_data_from_basic_stream(self, http_parameters: HttpxPostKwargs) -> AsyncGenerator[str, None]:
        async with self.async_client.stream('POST', **http_parameters) as source:
            async for line in source.aiter_lines():
                yield line
