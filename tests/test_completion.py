from __future__ import annotations

import asyncio
import time
from typing import Any, AsyncIterator, Iterator

from typing_extensions import Self

from generate.chat_completion import (
    ChatCompletionModel,
    ChatCompletionOutput,
    ChatCompletionStreamOutput,
    ModelParameters,
)
from generate.chat_completion.message import AssistantMessage, Messages, Prompts, UserMessage
from generate.chat_completion.model_output import Stream
from generate.completion_engine import CompletionEngine


class FakeChatParameters(ModelParameters):
    prefix: str = 'Completed:'


class FakeChat(ChatCompletionModel[FakeChatParameters]):
    model_type = 'test'

    def __init__(self, default_parameters: FakeChatParameters | None = None) -> None:
        default_parameters = default_parameters or FakeChatParameters()
        super().__init__(default_parameters)

    def _completion(self, messages: Messages, parameters: FakeChatParameters) -> ChatCompletionOutput:
        content = f'Completed: {messages[-1].content}'
        return ChatCompletionOutput(model_info=self.model_info, messages=[AssistantMessage(content=content)])

    async def _async_completion(self, messages: Messages, parameters: FakeChatParameters) -> ChatCompletionOutput:
        content = f'Completed: {messages[-1].content}'
        return ChatCompletionOutput(model_info=self.model_info, messages=[AssistantMessage(content=content)])

    def _stream_completion(self, messages: Messages, parameters: FakeChatParameters) -> Iterator[ChatCompletionStreamOutput]:
        content = f'Completed: {messages[-1].content}'
        yield ChatCompletionStreamOutput(
            model_info=self.model_info,
            messages=[AssistantMessage(content=content)],
            stream=Stream(delta=content, control='finish'),
        )

    async def _async_stream_completion(
        self, messages: Messages, parameters: FakeChatParameters
    ) -> AsyncIterator[ChatCompletionStreamOutput]:
        content = f'Completed: {messages[-1].content}'
        yield ChatCompletionStreamOutput(
            model_info=self.model_info,
            messages=[AssistantMessage(content=content)],
            stream=Stream(delta=content, control='finish'),
        )

    @property
    def name(self) -> str:
        return 'TestModel'

    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> Self:
        return cls()


def test_sync_completion() -> None:
    completion_model = FakeChat()
    client = CompletionEngine(completion_model)
    prompts = [
        'Hello, my name is',
        UserMessage(content='hello, who are you?'),
    ]
    results = list(client.run(prompts))

    assert isinstance(results[0].reply, str)
    assert results[0].reply == 'Completed: Hello, my name is'
    assert results[1].reply == 'Completed: hello, who are you?'
    assert len(results) == len(prompts)


async def async_helper(client: CompletionEngine, prompts: Prompts) -> list[ChatCompletionOutput]:
    return [result async for result in client.async_run(prompts)]


def test_async_completion() -> None:
    completion_model = FakeChat()
    client = CompletionEngine(completion_model, async_capacity=2, max_requests_per_minute=5)
    CompletionEngine.NUM_SECONDS_PER_MINUTE = 2

    start_time = time.perf_counter()
    messages = [{'role': 'user', 'content': 'hello, who are you?'}]
    prompts = ['Hello, my name is', 'I am a student', messages] * 4
    results = asyncio.run(async_helper(client, prompts))
    elapsed_time = time.perf_counter() - start_time

    assert results[0].reply == 'Completed: Hello, my name is'
    assert len(results) == len(prompts)
    assert elapsed_time > (2 * CompletionEngine.NUM_SECONDS_PER_MINUTE)
