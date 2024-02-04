from __future__ import annotations

import time

import anyio
from pydantic import BaseModel

from generate import OpenAIChat
from generate.chat_completion.model_output import ChatCompletionOutput
from generate.chat_completion.models.test import FakeChat


def test_limit() -> None:
    limited_model = FakeChat().limit(async_capacity=2, max_generates_per_time_window=5, num_seconds_in_time_window=2)

    start_time = time.perf_counter()
    messages = [{'role': 'user', 'content': 'hello, who are you?'}]
    prompts = ['Hello, my name is', 'I am a student', messages] * 4

    async def main() -> list[ChatCompletionOutput]:
        results = []
        async for i in limited_model.async_batch_generate(prompts):
            results.append(i)  # noqa
        return results

    results = anyio.run(main)
    elapsed_time = time.perf_counter() - start_time

    assert results[0].reply == 'Completed:Hello, my name is'
    assert len(results) == len(prompts)
    assert elapsed_time > (2 * limited_model.num_seconds_in_time_window)


def test_structure() -> None:
    class Country(BaseModel):
        name: str
        capital: str

    model = OpenAIChat().structure(instruction='提取文本中的国家实体', output_structure_type=Country)
    result = model.generate('Paris is the capital of France and also the largest city in the country.')
    assert result.structure == Country(name='France', capital='Paris')
