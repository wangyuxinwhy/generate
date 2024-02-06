from __future__ import annotations

from typing import AsyncIterator, ClassVar, Iterator

from typing_extensions import Self, Unpack, override

from generate.chat_completion.base import RemoteChatCompletionModel
from generate.chat_completion.message import Messages, Prompt, ensure_messages
from generate.chat_completion.model_output import ChatCompletionOutput, ChatCompletionStreamOutput
from generate.chat_completion.models.openai import (
    OpenAIChatParameters,
    OpenAIChatParametersDict,
    convert_to_openai_message,
    parse_openai_model_reponse,
)
from generate.http import HttpClient, HttpxPostKwargs
from generate.platforms.azure import AzureSettings


class AzureChat(RemoteChatCompletionModel):
    model_type: ClassVar[str] = 'azure'

    parameters: OpenAIChatParameters
    settings: AzureSettings

    def __init__(
        self,
        model: str,
        parameters: OpenAIChatParameters | None = None,
        settings: AzureSettings | None = None,
        http_client: HttpClient | None = None,
    ) -> None:
        parameters = parameters or OpenAIChatParameters()
        settings = settings or AzureSettings()  # type: ignore
        http_client = http_client or HttpClient()
        super().__init__(parameters=parameters, settings=settings, http_client=http_client)

        self.model = model

    def _get_request_parameters(self, messages: Messages, parameters: OpenAIChatParameters) -> HttpxPostKwargs:
        openai_messages = [convert_to_openai_message(message) for message in messages]
        json_data = {
            'model': self.model,
            'messages': openai_messages,
            **parameters.custom_model_dump(),
        }
        headers = {
            'api-key': self.settings.api_key.get_secret_value(),
        }
        return {
            'url': f'{self.settings.api_base}/openai/deployments/{self.model}/chat/completions',
            'headers': headers,
            'json': json_data,
            'params': {'api-version': self.settings.api_version},
        }

    @override
    def generate(self, prompt: Prompt, **kwargs: Unpack[OpenAIChatParametersDict]) -> ChatCompletionOutput:
        messages = ensure_messages(prompt)
        parameters = self.parameters.clone_with_changes(**kwargs)
        request_parameters = self._get_request_parameters(messages, parameters)
        response = self.http_client.post(request_parameters)
        output = parse_openai_model_reponse(response.json(), model_type=self.model_type)
        output.model_info.type = self.model_type
        return output

    @override
    async def async_generate(self, prompt: Prompt, **kwargs: Unpack[OpenAIChatParametersDict]) -> ChatCompletionOutput:
        messages = ensure_messages(prompt)
        parameters = self.parameters.clone_with_changes(**kwargs)
        request_parameters = self._get_request_parameters(messages, parameters)
        response = await self.http_client.async_post(request_parameters=request_parameters)
        output = parse_openai_model_reponse(response.json(), model_type=self.model_type)
        output.model_info.type = self.model_type
        return output

    @override
    def stream_generate(
        self, prompt: Prompt, **kwargs: Unpack[OpenAIChatParametersDict]
    ) -> Iterator[ChatCompletionStreamOutput]:
        raise NotImplementedError('Azure does not support streaming')

    @override
    def async_stream_generate(
        self, prompt: Prompt, **kwargs: Unpack[OpenAIChatParametersDict]
    ) -> AsyncIterator[ChatCompletionStreamOutput]:
        raise NotImplementedError('Azure does not support streaming')

    @property
    @override
    def name(self) -> str:
        return self.model

    @classmethod
    @override
    def from_name(cls, name: str) -> Self:
        return cls(model=name)
