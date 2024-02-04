from __future__ import annotations

import time
from typing import AsyncIterator, Iterator

import anyio
from typing_extensions import Self, Unpack

from generate.chat_completion import (
    ChatCompletionModel,
    ChatCompletionOutput,
    ChatCompletionStreamOutput,
)
from generate.chat_completion.message import AssistantMessage, Prompt, UserMessage, ensure_messages
from generate.chat_completion.model_output import Stream
from generate.limit import Limit
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
        return 'TestModel'

    @classmethod
    def from_name(cls, name: str) -> Self:
        return cls()


def test_sync_completion() -> None:
    completion_model = FakeChat()
    prompts = [
        'Hello, my name is',
        UserMessage(content='hello, who are you?'),
    ]
    results = list(completion_model.batch_generate(prompts))

    assert isinstance(results[0].reply, str)
    assert results[0].reply == 'Completed:Hello, my name is'
    assert results[1].reply == 'Completed:hello, who are you?'
    assert len(results) == len(prompts)


def test_async_completion() -> None:
    completion_model = FakeChat()
    limited_model = Limit(completion_model, async_capacity=2, max_requests_per_time_window=5, num_seconds_in_time_window=2)

    start_time = time.perf_counter()
    messages = [{'role': 'user', 'content': 'hello, who are you?'}]
    prompts = ['Hello, my name is', 'I am a student', messages] * 4

    async def main() -> list[ChatCompletionOutput]:
        results = []
        async for i in limited_model.async_batch_generate(prompts):
            results.append(i) # noqa
        return results

    results = anyio.run(main)
    elapsed_time = time.perf_counter() - start_time

    assert results[0].reply == 'Completed:Hello, my name is'
    assert len(results) == len(prompts)
    assert elapsed_time > (2 * limited_model.num_seconds_in_time_window)
