from __future__ import annotations

from typing import ClassVar, List

from generate.chat_completion.message.converter import MessageConverter
from generate.chat_completion.models.openai_like import (
    OpenAIChatParameters,
    OpenAILikeChat,
    SupportOpenAIToolCall,
)
from generate.http import (
    HttpClient,
)
from generate.platforms.openai import OpenAISettings


class OpenAIChat(OpenAILikeChat, SupportOpenAIToolCall):
    model_type: ClassVar[str] = 'openai'
    available_models: ClassVar[List[str]] = [
        'gpt-4-turbo-preview',
        'gpt-3.5-turbo',
        'gpt-4-vision-preview',
    ]

    settings: OpenAISettings
    parameters: OpenAIChatParameters

    def __init__(
        self,
        model: str = 'gpt-3.5-turbo',
        parameters: OpenAIChatParameters | None = None,
        settings: OpenAISettings | None = None,
        http_client: HttpClient | None = None,
        message_converter: MessageConverter | None = None,
    ) -> None:
        parameters = parameters or OpenAIChatParameters()
        settings = settings or OpenAISettings()  # type: ignore
        http_client = http_client or HttpClient()
        super().__init__(
            model=model,
            parameters=parameters,
            settings=settings,
            message_converter=message_converter,
            http_client=http_client,
        )
