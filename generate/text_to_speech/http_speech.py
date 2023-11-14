from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypeVar

import httpx
from httpx import Response
from tenacity import retry, stop_after_attempt, wait_random_exponential
from typing_extensions import override

from generate.http import HttpModelInitKwargs as HttpModelInitKwargs
from generate.http import HttpxPostKwargs, ProxiesTypes, RetryStrategy
from generate.parameters import ModelParameters
from generate.text_to_speech.base import TextToSpeechModel
from generate.text_to_speech.model_output import TextToSpeechModelOutput

P = TypeVar('P', bound=ModelParameters)


class HttpSpeechModel(TextToSpeechModel[P], ABC):
    model_type = 'http'

    def __init__(
        self,
        parameters: P,
        timeout: int | None = None,
        retry: bool | RetryStrategy = False,
        proxies: ProxiesTypes | None = None,
    ) -> None:
        super().__init__(parameters=parameters)
        self.timeout = timeout or 60
        if isinstance(retry, RetryStrategy):
            self.retry_strategy = retry
        else:
            self.retry_strategy = RetryStrategy() if retry else None
        self.proxies = proxies

    @abstractmethod
    def _get_request_parameters(self, text: str, parameters: P) -> HttpxPostKwargs:
        ...

    @abstractmethod
    def _construct_model_output(self, text: str, parameters: P, response: Response) -> TextToSpeechModelOutput:
        ...

    def _completion_without_retry(self, text: str, parameters: P) -> TextToSpeechModelOutput:
        with httpx.Client(proxies=self.proxies) as client:
            http_parameters = self._get_request_parameters(text, parameters)
            http_parameters.update({'timeout': self.timeout})
            http_response = client.post(**http_parameters)  # type: ignore
        http_response.raise_for_status()
        model_output = self._construct_model_output(text, parameters, http_response)
        http_parameters['headers'].pop('Authorization', None)
        model_output.debug['http_request'] = http_parameters
        return model_output

    async def _async_completion_without_retry(self, text: str, parameters: P) -> TextToSpeechModelOutput:
        async with httpx.AsyncClient(proxies=self.proxies) as client:
            http_parameters = self._get_request_parameters(text, parameters)
            http_parameters.update({'timeout': self.timeout})
            http_response = await client.post(**http_parameters)  # type: ignore
        http_response.raise_for_status()
        model_output = self._construct_model_output(text, parameters, http_response)
        http_parameters['headers'].pop('Authorization', None)
        model_output.debug['http_request'] = http_parameters
        return model_output

    @override
    def _generate(self, text: str, parameters: P) -> TextToSpeechModelOutput:
        if self.retry_strategy is None:
            return self._completion_without_retry(text, parameters)

        wait = wait_random_exponential(min=self.retry_strategy.min_wait_seconds, max=self.retry_strategy.max_wait_seconds)
        stop = stop_after_attempt(self.retry_strategy.max_attempt)
        return retry(wait=wait, stop=stop)(self._completion_without_retry)(text, parameters)

    @override
    async def _async_generate(self, text: str, parameters: P) -> TextToSpeechModelOutput:
        if self.retry_strategy is None:
            return await self._async_completion_without_retry(text, parameters)

        wait = wait_random_exponential(min=self.retry_strategy.min_wait_seconds, max=self.retry_strategy.max_wait_seconds)
        stop = stop_after_attempt(self.retry_strategy.max_attempt)
        return await retry(wait=wait, stop=stop)(self._async_completion_without_retry)(text, parameters)
