from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, ClassVar, Generic, Iterator, TypeVar

from typing_extensions import Self, TypeGuard

from generate.chat_completion.message import Messages, Prompt, ensure_messages
from generate.chat_completion.model_output import ChatCompletionModelOutput, ChatCompletionModelStreamOutput
from generate.parameters import ModelParameters

P = TypeVar('P', bound=ModelParameters)


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
    def _completion(self, messages: Messages, parameters: P) -> ChatCompletionModelOutput:
        ...

    @abstractmethod
    async def _async_completion(self, messages: Messages, parameters: P) -> ChatCompletionModelOutput:
        ...

    @abstractmethod
    def _stream_completion(self, messages: Messages, parameters: P) -> Iterator[ChatCompletionModelStreamOutput]:
        ...

    @abstractmethod
    def _async_stream_completion(self, messages: Messages, parameters: P) -> AsyncIterator[ChatCompletionModelStreamOutput]:
        ...

    @property
    def model_id(self) -> str:
        return f'{self.model_type}/{self.name}'

    def generate(self, prompt: Prompt, **override_parameters: Any) -> ChatCompletionModelOutput:
        parameters = self._merge_parameters(**override_parameters)
        messages = ensure_messages(prompt)
        model_output = self._completion(messages, parameters)
        model_output.debug['input_messages'] = messages
        model_output.debug['parameters'] = parameters
        return model_output

    async def async_generate(self, prompt: Prompt, **override_parameters: Any) -> ChatCompletionModelOutput:
        parameters = self._merge_parameters(**override_parameters)
        messages = ensure_messages(prompt)
        model_output = await self._async_completion(messages, parameters)
        model_output.debug['input_messages'] = messages
        model_output.debug['parameters'] = parameters
        return model_output

    def stream_generate(self, prompt: Prompt, **override_parameters: Any) -> Iterator[ChatCompletionModelStreamOutput]:
        parameters = self._merge_parameters(**override_parameters)
        messages = ensure_messages(prompt)
        for stream_output in self._stream_completion(messages, parameters):
            if stream_output.is_finish:
                stream_output.debug['input_messages'] = messages
                stream_output.debug['parameters'] = parameters
            yield stream_output

    async def async_stream_generate(
        self, prompt: Prompt, **override_parameters: Any
    ) -> AsyncIterator[ChatCompletionModelStreamOutput]:
        parameters = self._merge_parameters(**override_parameters)
        messages = ensure_messages(prompt)
        async for stream_output in self._async_stream_completion(messages, parameters):
            if stream_output.is_finish:
                stream_output.debug['input_messages'] = messages
                stream_output.debug['parameters'] = parameters
            yield stream_output

    def _merge_parameters(self, **override_parameters: Any) -> P:
        return self.parameters.__class__.model_validate(
            {**self.parameters.model_dump(exclude_unset=True), **override_parameters}
        )


def is_stream_model_output(model_output: ChatCompletionModelOutput) -> TypeGuard[ChatCompletionModelStreamOutput]:
    return getattr(model_output, 'stream', None) is not None
