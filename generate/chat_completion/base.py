from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, ClassVar, Iterator, TypeVar

from generate.chat_completion.message import Messages, Prompt, ensure_messages
from generate.chat_completion.model_output import ChatCompletionOutput, ChatCompletionStreamOutput
from generate.model import GenerateModel, ModelParameters

P = TypeVar('P', bound=ModelParameters)
logger = logging.getLogger(__name__)


class ChatCompletionModel(GenerateModel[P, Prompt, ChatCompletionOutput], ABC):
    model_task: ClassVar[str] = 'chat_completion'
    model_type: ClassVar[str]

    @abstractmethod
    def _completion(self, messages: Messages, parameters: P) -> ChatCompletionOutput:
        ...

    @abstractmethod
    async def _async_completion(self, messages: Messages, parameters: P) -> ChatCompletionOutput:
        ...

    @abstractmethod
    def _stream_completion(self, messages: Messages, parameters: P) -> Iterator[ChatCompletionStreamOutput]:
        ...

    @abstractmethod
    def _async_stream_completion(self, messages: Messages, parameters: P) -> AsyncIterator[ChatCompletionStreamOutput]:
        ...

    def generate(self, prompt: Prompt, **override_parameters: Any) -> ChatCompletionOutput:
        parameters = self._merge_parameters(**override_parameters)
        messages = ensure_messages(prompt)
        logger.debug(f'{messages=}, {parameters=}')
        return self._completion(messages, parameters)

    async def async_generate(self, prompt: Prompt, **override_parameters: Any) -> ChatCompletionOutput:
        parameters = self._merge_parameters(**override_parameters)
        messages = ensure_messages(prompt)
        logger.debug(f'{messages=}, {parameters=}')
        return await self._async_completion(messages, parameters)

    def stream_generate(self, prompt: Prompt, **override_parameters: Any) -> Iterator[ChatCompletionStreamOutput]:
        parameters = self._merge_parameters(**override_parameters)
        messages = ensure_messages(prompt)
        logger.debug(f'{messages=}, {parameters=}')
        return self._stream_completion(messages, parameters)

    def async_stream_generate(self, prompt: Prompt, **override_parameters: Any) -> AsyncIterator[ChatCompletionStreamOutput]:
        parameters = self._merge_parameters(**override_parameters)
        messages = ensure_messages(prompt)
        logger.debug(f'{messages=}, {parameters=}')
        return self._async_stream_completion(messages, parameters)
