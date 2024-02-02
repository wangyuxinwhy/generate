from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import AsyncIterator, ClassVar, Iterator

from pydantic_settings import BaseSettings
from typing_extensions import Unpack

from generate.chat_completion.message import Prompt
from generate.chat_completion.model_output import ChatCompletionOutput, ChatCompletionStreamOutput
from generate.http import HttpClient
from generate.model import GenerateModel, ModelParameters, ModelParametersDict

logger = logging.getLogger(__name__)


class ChatCompletionModel(GenerateModel[Prompt, ChatCompletionOutput], ABC):
    model_task: ClassVar[str] = 'chat_completion'
    model_type: ClassVar[str]

    @abstractmethod
    def stream_generate(self, prompt: Prompt, **kwargs: Unpack[ModelParametersDict]) -> Iterator[ChatCompletionStreamOutput]:
        ...

    @abstractmethod
    def async_stream_generate(
        self, prompt: Prompt, **kwargs: Unpack[ModelParametersDict]
    ) -> AsyncIterator[ChatCompletionStreamOutput]:
        ...


class RemoteChatCompletionModel(ChatCompletionModel):
    def __init__(
        self,
        parameters: ModelParameters,
        settings: BaseSettings,
        http_client: HttpClient,
    ) -> None:
        self.parameters = parameters
        self.settings = settings
        self.http_client = http_client
