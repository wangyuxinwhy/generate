from __future__ import annotations

from typing import Any, AsyncIterator, ClassVar, Iterator

from typing_extensions import Unpack, override

from generate.chat_completion.base import RemoteChatCompletionModel
from generate.chat_completion.message import Prompt, ensure_messages
from generate.chat_completion.model_output import ChatCompletionOutput, ChatCompletionStreamOutput
from generate.chat_completion.models.openai import OpenAIChatParameters, OpenAIChatParametersDict
from generate.chat_completion.models.openai_like import convert_to_openai_message, process_openai_like_model_reponse
from generate.chat_completion.stream_manager import StreamManager
from generate.http import HttpClient, HttpxPostKwargs
from generate.platforms.azure import AzureSettings


class AzureChat(RemoteChatCompletionModel):
    model_type: ClassVar[str] = 'azure'

    parameters: OpenAIChatParameters
    settings: AzureSettings

    def __init__(
        self,
        model: str | None = None,
        parameters: OpenAIChatParameters | None = None,
        settings: AzureSettings | None = None,
        http_client: HttpClient | None = None,
    ) -> None:
        parameters = parameters or OpenAIChatParameters()
        settings = settings or AzureSettings()  # type: ignore
        http_client = http_client or HttpClient()
        model = model or settings.chat_api_engine
        if model is None:
            raise ValueError('model must be provided or set in settings.chat_api_engine')
        super().__init__(model, parameters=parameters, settings=settings, http_client=http_client)

    @override
    def generate(self, prompt: Prompt, **kwargs: Unpack[OpenAIChatParametersDict]) -> ChatCompletionOutput:
        return super().generate(prompt, **kwargs)

    @override
    async def async_generate(self, prompt: Prompt, **kwargs: Unpack[OpenAIChatParametersDict]) -> ChatCompletionOutput:
        return await super().async_generate(prompt, **kwargs)

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

    @override
    def _get_request_parameters(self, prompt: Prompt, **kwargs: Unpack[OpenAIChatParametersDict]) -> HttpxPostKwargs:
        messages = ensure_messages(prompt)
        parameters = self.parameters.clone_with_changes(**kwargs)

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
    def _process_reponse(self, response: dict[str, Any]) -> ChatCompletionOutput:
        return process_openai_like_model_reponse(response, model_type=self.model_type)

    @override
    def _process_stream_line(self, line: str, stream_manager: StreamManager) -> ChatCompletionStreamOutput | None:
        raise NotImplementedError
