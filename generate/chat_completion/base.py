from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, ClassVar, Generic, Iterator, TypeVar

from typing_extensions import Self, TypeGuard

from generate.chat_completion.message import Messages, Prompt, ensure_messages
from generate.chat_completion.model_output import ChatCompletionOutput, ChatCompletionStreamOutput
from generate.model import ModelInfo
from generate.parameters import ModelParameters

P = TypeVar('P', bound=ModelParameters)
logger = logging.getLogger(__name__)


class ChatCompletionModel(Generic[P], ABC):
    model_type: ClassVar[str]

    def __init__(self, parameters: P) -> None:
        self.parameters = parameters

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @classmethod
    @abstractmethod
    def from_name(cls, name: str, **kwargs: Any) -> Self:
        ...

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

    @property
    def model_info(self) -> ModelInfo:
        return ModelInfo(task='chat_completion', type=self.model_type, name=self.name)

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

    def _merge_parameters(self, **override_parameters: Any) -> P:
        return self.parameters.__class__.model_validate(
            {**self.parameters.model_dump(exclude_unset=True), **override_parameters}
        )


def is_stream_model_output(model_output: ChatCompletionOutput) -> TypeGuard[ChatCompletionStreamOutput]:
    return getattr(model_output, 'stream', None) is not None
