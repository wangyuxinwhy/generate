from __future__ import annotations

import os
from typing import Any, AsyncIterator, ClassVar, Iterator

from typing_extensions import Self, Unpack, override

from generate.chat_completion.base import ChatCompletionModel
from generate.chat_completion.message import Messages
from generate.chat_completion.model_output import ChatCompletionOutput, ChatCompletionStreamOutput
from generate.chat_completion.models.openai import (
    OpenAIChatParameters,
    convert_to_openai_message,
    parse_openai_model_reponse,
)
from generate.http import HttpClient, HttpClientInitKwargs, HttpMixin, HttpResponse, HttpxPostKwargs


class AzureChat(ChatCompletionModel[OpenAIChatParameters], HttpMixin):
    model_type: ClassVar[str] = 'azure'

    def __init__(
        self,
        model: str | None = None,
        system_prompt: str | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        api_version: str | None = None,
        parameters: OpenAIChatParameters | None = None,
        **kwargs: Unpack[HttpClientInitKwargs],
    ) -> None:
        parameters = parameters or OpenAIChatParameters()
        super().__init__(parameters=parameters)
        self.model = model or os.environ['AZURE_CHAT_API_ENGINE'] or os.environ['AZURE_CHAT_MODEL_NAME']
        self.system_prompt = system_prompt
        self.api_key = api_key or os.environ['AZURE_API_KEY']
        self.api_base = api_base or os.environ['AZURE_API_BASE']
        self.api_version = api_version or os.getenv('AZURE_API_VERSION')
        self.http_client = HttpClient(**kwargs)

    def _get_request_parameters(self, messages: Messages, parameters: OpenAIChatParameters) -> HttpxPostKwargs:
        openai_messages = [convert_to_openai_message(message) for message in messages]
        if self.system_prompt:
            openai_messages.insert(0, {'role': 'system', 'content': self.system_prompt})

        json_data = {
            'model': self.model,
            'messages': openai_messages,
            **parameters.custom_model_dump(),
        }
        headers = {
            'api-key': self.api_key,
        }
        return {
            'url': f'{self.api_base}/openai/deployments/{self.model}/chat/completions?api-version={self.api_version}',
            'headers': headers,
            'json': json_data,
        }

    def _completion(self, messages: Messages, parameters: OpenAIChatParameters) -> ChatCompletionOutput:
        request_parameters = self._get_request_parameters(messages, parameters)
        response = self.http_client.post(request_parameters=request_parameters)
        output = parse_openai_model_reponse(response.json())
        output.model_info.type = self.model_type
        return output

    async def _async_completion(self, messages: Messages, parameters: OpenAIChatParameters) -> ChatCompletionOutput:
        request_parameters = self._get_request_parameters(messages, parameters)
        response = await self.http_client.async_post(request_parameters=request_parameters)
        output = parse_openai_model_reponse(response.json())
        output.model_info.type = self.model_type
        return output

    def _parse_reponse(self, response: HttpResponse) -> ChatCompletionOutput:
        return parse_openai_model_reponse(response)

    @override
    def _stream_completion(self, messages: Messages, parameters: OpenAIChatParameters) -> Iterator[ChatCompletionStreamOutput]:
        raise NotImplementedError('Azure does not support streaming')

    @override
    async def _async_stream_completion(
        self, messages: Messages, parameters: OpenAIChatParameters
    ) -> AsyncIterator[ChatCompletionStreamOutput]:
        raise NotImplementedError('Azure does not support streaming')

    @property
    @override
    def name(self) -> str:
        return self.model

    @classmethod
    @override
    def from_name(cls, name: str, **kwargs: Any) -> Self:
        return cls(model=name, **kwargs)
