from __future__ import annotations

import anyio

from generate.chat_completion.message import UserMessage
from generate.chat_completion.model_output import ChatCompletionOutput
from generate.chat_completion.models.test import FakeChat


def test_sync_completion() -> None:
    completion_model = FakeChat()
    prompts = [
        'Hello, my name is',
        UserMessage(content='hello, who are you?'),
    ]
    result = completion_model.generate(prompts[0])
    assert result.reply == 'Completed:Hello, my name is'

    results = list(completion_model.batch_generate(prompts))
    assert results[0].reply == 'Completed:Hello, my name is'
    assert results[1].reply == 'Completed:hello, who are you?'
    assert len(results) == len(prompts)


def test_async_generate() -> None:
    completion_model = FakeChat()
    messages = [{'role': 'user', 'content': 'hello, who are you?'}]
    prompts = ['Hello, my name is', 'I am a student', messages] * 4

    async def generate() -> ChatCompletionOutput:
        return await completion_model.async_generate(prompts[0])

    result = anyio.run(generate)
    assert result.reply == 'Completed:Hello, my name is'

    async def batch_generate() -> list[ChatCompletionOutput]:
        return [result async for result in completion_model.async_batch_generate(prompts)]

    results = anyio.run(batch_generate)

    assert results[0].reply == 'Completed:Hello, my name is'
    assert len(results) == len(prompts)
