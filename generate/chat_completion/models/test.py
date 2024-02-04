from __future__ import annotations

from typing import AsyncIterator, Iterator

from typing_extensions import Self, Unpack

from generate.chat_completion import (
    ChatCompletionModel,
    ChatCompletionOutput,
    ChatCompletionStreamOutput,
)
from generate.chat_completion.message import AssistantMessage, Prompt, ensure_messages
from generate.chat_completion.model_output import Stream
from generate.model import ModelParameters, ModelParametersDict


class FakeChatParameters(ModelParameters):
    prefix: str = 'Completed:'


class FakeChatParametersDict(ModelParametersDict, total=False):
    prefix: str


class FakeChat(ChatCompletionModel):
    model_type = 'test'

    def __init__(self, parameters: FakeChatParameters | None = None) -> None:
        self.parameters = parameters or FakeChatParameters()

    def generate(self, prompt: Prompt, **kwargs: Unpack[ModelParametersDict]) -> ChatCompletionOutput:
        messages = ensure_messages(prompt)
        parameters = self.parameters.clone_with_changes(**kwargs)
        content = f'{parameters.prefix}{messages[-1].content}'
        return ChatCompletionOutput(model_info=self.model_info, message=AssistantMessage(content=content))

    async def async_generate(self, prompt: Prompt, **kwargs: Unpack[ModelParametersDict]) -> ChatCompletionOutput:
        messages = ensure_messages(prompt)
        parameters = self.parameters.clone_with_changes(**kwargs)
        content = f'{parameters.prefix}{messages[-1].content}'
        return ChatCompletionOutput(model_info=self.model_info, message=AssistantMessage(content=content))

    def stream_generate(self, prompt: Prompt, **kwargs: Unpack[ModelParametersDict]) -> Iterator[ChatCompletionStreamOutput]:
        messages = ensure_messages(prompt)
        parameters = self.parameters.clone_with_changes(**kwargs)
        content = f'{parameters.prefix}{messages[-1].content}'
        yield ChatCompletionStreamOutput(
            model_info=self.model_info,
            message=AssistantMessage(content=''),
            stream=Stream(delta='', control='start'),
        )
        yield ChatCompletionStreamOutput(
            model_info=self.model_info,
            message=AssistantMessage(content=content),
            stream=Stream(delta=content, control='finish'),
        )

    async def async_stream_generate(
        self, prompt: Prompt, **kwargs: Unpack[ModelParametersDict]
    ) -> AsyncIterator[ChatCompletionStreamOutput]:
        messages = ensure_messages(prompt)
        parameters = self.parameters.clone_with_changes(**kwargs)
        content = f'{parameters.prefix}{messages[-1].content}'
        yield ChatCompletionStreamOutput(
            model_info=self.model_info,
            message=AssistantMessage(content=''),
            stream=Stream(delta='', control='start'),
        )
        yield ChatCompletionStreamOutput(
            model_info=self.model_info,
            message=AssistantMessage(content=content),
            stream=Stream(delta=content, control='finish'),
        )

    @property
    def name(self) -> str:
        return 'Fake'

    @classmethod
    def from_name(cls, name: str) -> Self:
        return cls()
