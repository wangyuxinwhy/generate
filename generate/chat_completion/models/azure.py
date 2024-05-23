from __future__ import annotations

from typing import Any, AsyncIterator, ClassVar, Dict, Iterator

from typing_extensions import Unpack, override

from generate.chat_completion.message import Prompt
from generate.chat_completion.message.core import Messages
from generate.chat_completion.model_output import ChatCompletionStreamOutput
from generate.chat_completion.models.openai_like import (
    OpenAIChatParameters,
    OpenAIChatParametersDict,
    OpenAILikeChat,
    OpenAIMessageConverter,
)
from generate.chat_completion.stream_manager import StreamManager
from generate.http import HttpClient, HttpxPostKwargs
from generate.platforms.azure import AzureSettings


class AzureChat(OpenAILikeChat):
    model_type: ClassVar[str] = 'azure'

    parameters: OpenAIChatParameters
    settings: AzureSettings
    message_converter: OpenAIMessageConverter

    def __init__(
        self,
        model: str | None = None,
        parameters: OpenAIChatParameters | None = None,
        settings: AzureSettings | None = None,
        http_client: HttpClient | None = None,
        message_converter: OpenAIMessageConverter | None = None,
    ) -> None:
        parameters = parameters or OpenAIChatParameters()
        settings = settings or AzureSettings()  # type: ignore
        http_client = http_client or HttpClient()
        model = model or settings.chat_api_engine
        if model is None:
            raise ValueError('model must be provided or set in settings.chat_api_engine')
        message_converter = message_converter or OpenAIMessageConverter()
        super().__init__(
            model=model, parameters=parameters, settings=settings, http_client=http_client, message_converter=message_converter
        )

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
    def _get_request_parameters(self, messages: Messages, **kwargs: Unpack[OpenAIChatParametersDict]) -> HttpxPostKwargs:
        parameters = self.parameters.clone_with_changes(**kwargs)
        json_data = {
            'model': self.model,
            'messages': self.message_converter.convert_messages(messages),
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
    def _process_stream_response(
        self, response: Dict[str, Any], stream_manager: StreamManager
    ) -> ChatCompletionStreamOutput | None:
        raise NotImplementedError('Azure does not support streaming')
