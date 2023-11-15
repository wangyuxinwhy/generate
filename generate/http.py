from __future__ import annotations

import logging
from typing import Any, AsyncGenerator, Dict, Generator, Literal, Mapping, Optional, Sequence, Union

import httpx
from httpx import Response
from httpx._types import ProxiesTypes
from httpx_sse import aconnect_sse, connect_sse
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_random_exponential
from typing_extensions import Required, TypedDict

from generate.types import PrimitiveData

logger = logging.getLogger(__name__)
HttpResponse = Dict[str, Any]
QueryParams = Mapping[str, Union[PrimitiveData, Sequence[PrimitiveData]]]
Headers = Dict[str, str]


class RetryStrategy(BaseModel):
    min_wait_seconds: int = 2
    max_wait_seconds: int = 20
    max_attempt: int = 3


class HttpClientInitKwargs(TypedDict, total=False):
    timeout: Optional[int]
    retry: Union[bool, RetryStrategy]
    proxies: Union[ProxiesTypes, None]


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
        response (dict): The response from the server.
    """

    def __init__(self, response: dict, *args: Any) -> None:
        super().__init__(response, *args)


class HttpClient:
    def __init__(
        self,
        retry: Union[bool, RetryStrategy] = False,
        timeout: int | None = None,
        proxies: ProxiesTypes | None = None,
    ) -> None:
        self.timeout = timeout or 60
        if isinstance(retry, RetryStrategy):
            self.retry_strategy = retry
        else:
            self.retry_strategy = RetryStrategy() if retry else None
        self.proxies = proxies

    def get(self, request_parameters: HttpGetKwargs) -> Response:
        if self.retry_strategy is None:
            return self._get(request_parameters)

        wait = wait_random_exponential(min=self.retry_strategy.min_wait_seconds, max=self.retry_strategy.max_wait_seconds)
        stop = stop_after_attempt(self.retry_strategy.max_attempt)
        return retry(wait=wait, stop=stop)(self._get)(request_parameters)

    async def async_get(self, request_parameters: HttpGetKwargs) -> Response:
        if self.retry_strategy is None:
            return await self._async_get(request_parameters)

        wait = wait_random_exponential(min=self.retry_strategy.min_wait_seconds, max=self.retry_strategy.max_wait_seconds)
        stop = stop_after_attempt(self.retry_strategy.max_attempt)
        return await retry(wait=wait, stop=stop)(self._async_get)(request_parameters)

    def post(self, request_parameters: HttpxPostKwargs) -> Response:
        if self.retry_strategy is None:
            return self._post(request_parameters)

        wait = wait_random_exponential(min=self.retry_strategy.min_wait_seconds, max=self.retry_strategy.max_wait_seconds)
        stop = stop_after_attempt(self.retry_strategy.max_attempt)
        return retry(wait=wait, stop=stop)(self._post)(request_parameters)

    async def async_post(self, request_parameters: HttpxPostKwargs) -> Response:
        if self.retry_strategy is None:
            return await self._async_post(request_parameters)

        wait = wait_random_exponential(min=self.retry_strategy.min_wait_seconds, max=self.retry_strategy.max_wait_seconds)
        stop = stop_after_attempt(self.retry_strategy.max_attempt)
        return await retry(wait=wait, stop=stop)(self._async_post)(request_parameters)

    def _post(self, request_parameters: HttpxPostKwargs) -> Response:
        with httpx.Client(proxies=self.proxies) as client:
            request_parameters.update({'timeout': self.timeout})
            logger.debug(f'POST {request_parameters}')
            http_response = client.post(**request_parameters)  # type: ignore
        http_response.raise_for_status()
        logger.debug(f'Response {http_response}')
        return http_response

    async def _async_post(self, request_parameters: HttpxPostKwargs) -> Response:
        async with httpx.AsyncClient(proxies=self.proxies) as client:
            request_parameters.update({'timeout': self.timeout})
            logger.debug(f'POST {request_parameters}')
            http_response = await client.post(**request_parameters)  # type: ignore
        http_response.raise_for_status()
        logger.debug(f'Response {http_response}')
        return http_response

    def _get(self, request_parameters: HttpGetKwargs) -> Response:
        with httpx.Client(proxies=self.proxies) as client:
            request_parameters.update({'timeout': self.timeout})
            logger.debug(f'GET {request_parameters}')
            http_response = client.get(**request_parameters)
        http_response.raise_for_status()
        return http_response

    async def _async_get(self, request_parameters: HttpGetKwargs) -> Response:
        async with httpx.AsyncClient(proxies=self.proxies) as client:
            request_parameters.update({'timeout': self.timeout})
            logger.debug(f'GET {request_parameters}')
            http_response = await client.get(**request_parameters)
        http_response.raise_for_status()
        return http_response


class HttpStreamClient:
    def __init__(
        self,
        retry: Union[bool, RetryStrategy] = False,
        timeout: int | None = None,
        proxies: ProxiesTypes | None = None,
        stream_strategy: Literal['sse', 'basic'] = 'sse',
    ) -> None:
        self.timeout = timeout or 60
        if isinstance(retry, RetryStrategy):
            self.retry_strategy = retry
        else:
            self.retry_strategy = RetryStrategy() if retry else None
        self.proxies = proxies
        self.stream_strategy = stream_strategy

    def post(self, request_parameters: HttpxPostKwargs) -> Generator[str, None, None]:
        if self.retry_strategy is None:
            return self._post(request_parameters)

        wait = wait_random_exponential(min=self.retry_strategy.min_wait_seconds, max=self.retry_strategy.max_wait_seconds)
        stop = stop_after_attempt(self.retry_strategy.max_attempt)
        return retry(wait=wait, stop=stop)(self._post)(request_parameters)

    def async_post(self, request_parameters: HttpxPostKwargs) -> AsyncGenerator[str, None]:
        if self.retry_strategy is None:
            return self._async_post(request_parameters)

        wait = wait_random_exponential(min=self.retry_strategy.min_wait_seconds, max=self.retry_strategy.max_wait_seconds)
        stop = stop_after_attempt(self.retry_strategy.max_attempt)
        return retry(wait=wait, stop=stop)(self._async_post)(request_parameters)

    def _post(self, request_parameters: HttpxPostKwargs) -> Generator[str, None, None]:
        request_parameters.update({'timeout': self.timeout})
        logger.debug(f'POST {request_parameters}')
        if self.stream_strategy == 'sse':
            return self._generate_data_from_sse_stream(request_parameters)
        return self._generate_data_from_basic_stream(request_parameters)

    def _generate_data_from_sse_stream(self, request_parameters: HttpxPostKwargs) -> Generator[str, None, None]:
        with httpx.Client(proxies=self.proxies) as client, connect_sse(
            client=client, method='POST', **request_parameters
        ) as source:
            for sse in source.iter_sse():
                yield sse.data

    def _generate_data_from_basic_stream(self, request_parameters: HttpxPostKwargs) -> Generator[str, None, None]:
        with httpx.Client(proxies=self.proxies) as client, client.stream('POST', **request_parameters) as source:
            for line in source.iter_lines():
                yield line

    def _async_post(self, request_parameters: HttpxPostKwargs) -> AsyncGenerator[str, None]:
        request_parameters.update({'timeout': self.timeout})
        logger.debug(f'POST {request_parameters}')
        if self.stream_strategy == 'sse':
            return self._async_generate_data_from_sse_stream(request_parameters)
        return self._async_generate_data_from_basic_stream(request_parameters)

    async def _async_generate_data_from_sse_stream(self, http_parameters: HttpxPostKwargs) -> AsyncGenerator[str, None]:
        async with httpx.AsyncClient(proxies=self.proxies) as client, aconnect_sse(
            client=client, method='POST', **http_parameters
        ) as source:
            async for sse in source.aiter_sse():
                yield sse.data

    async def _async_generate_data_from_basic_stream(self, http_parameters: HttpxPostKwargs) -> AsyncGenerator[str, None]:
        async with httpx.AsyncClient(proxies=self.proxies) as client, client.stream('POST', **http_parameters) as source:
            async for line in source.aiter_lines():
                yield line


class HttpMixin:
    http_client: HttpClient

    @property
    def timeout(self) -> int:
        return self.http_client.timeout

    @timeout.setter
    def timeout(self, timeout: int) -> None:
        self.http_client.timeout = timeout

    @property
    def retry(self) -> RetryStrategy | None:
        return self.http_client.retry_strategy

    @retry.setter
    def retry(self, retry: Union[bool, RetryStrategy]) -> None:
        if isinstance(retry, RetryStrategy):
            self.retry_strategy = retry
        else:
            self.retry_strategy = RetryStrategy() if retry else None

    @property
    def proxies(self) -> ProxiesTypes | None:
        return self.http_client.proxies

    @proxies.setter
    def proxies(self, proxies: ProxiesTypes | None) -> None:
        self.http_client.proxies = proxies
