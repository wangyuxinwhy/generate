from __future__ import annotations

import time
from typing import (
    AsyncGenerator,
    ClassVar,
    Generator,
    Literal,
    NoReturn,
)

import anyio
import asyncer
import tqdm
from typing_extensions import Self, TypedDict, Unpack

from generate.chat_completion import ChatCompletionModel, ChatCompletionOutput
from generate.chat_completion.message import AssistantMessage, Prompt, Prompts, ensure_messages
from generate.utils import load_chat_model


class CompletionEngineKwargs(TypedDict):
    async_capacity: int
    max_requests_per_minute: int
    error_mode: Literal['raise', 'ignore']
    progress_bar_mode: Literal['auto', 'never', 'always']


class CompletionEngine:
    """
    This is the completion engine for the chat completion model.

    Responsible for managing and executing large-scale requests to a given chat completion model, this class introduces
    capabilities like asynchronous requests handling, rate limiting (max requests per minute), and progress tracking.

    Args:
        chat_model (ChatCompletionModel): The chat model for generating completions.
        async_capacity (int, optional): The maximum number of asynchronous requests that can be made concurrently. Defaults to 3.
        max_requests_per_minute (int, optional): The maximum number of requests that can be made per minute to avoid rate limiting. Defaults to 20.
        error_mode (Literal['raise', 'ignore'], optional): The error handling mode. If 'raise', it raises the exception, if 'ignore', it ignores the exception and returns an error message. Defaults to 'raise'.
        progress_bar_mode (Literal['auto', 'never', 'always'], optional): The progress bar mode. If 'always', it always shows the progress bar, if 'auto', it shows the progress bar when the number of tasks exceeds a certain threshold. Defaults to 'auto'.
    """

    NUM_SECONDS_PER_MINUTE: ClassVar[int] = 60
    PROGRESS_BAR_THRESHOLD: ClassVar[int] = 20

    def __init__(
        self,
        chat_model: ChatCompletionModel,
        async_capacity: int = 3,
        max_requests_per_minute: int = 20,
        error_mode: Literal['raise', 'ignore'] = 'raise',
        progress_bar_mode: Literal['auto', 'never', 'always'] = 'auto',
    ) -> None:
        self.chat_model = chat_model
        self.async_capacity = async_capacity
        self.max_requests_per_minute = max_requests_per_minute
        self.error_mode = error_mode
        self.progress_bar_mode = progress_bar_mode
        self._task_created_time_list: list[int] = []

    @classmethod
    def from_model_id(cls, model_id: str, **kwargs: Unpack[CompletionEngineKwargs]) -> Self:
        chat_model = load_chat_model(model_id)
        return cls(chat_model, **kwargs)

    def run(self, prompts: Prompts) -> Generator[ChatCompletionOutput, None, None]:
        progress_bar = self._get_progress_bar(num_tasks=len(prompts))
        for prompt in prompts:
            task_result = self._run_single_task(prompt=prompt, progress_bar=progress_bar)
            yield task_result
        progress_bar.close()

    def _run_single_task(
        self,
        prompt: Prompt,
        progress_bar: tqdm.tqdm[NoReturn],
    ) -> ChatCompletionOutput:
        messages = ensure_messages(prompt)
        sleep_time = self._calculate_sleep_time()
        if sleep_time > 0:
            time.sleep(sleep_time)
        self._task_created_time_list.append(int(time.time()))

        try:
            output = self.chat_model.generate(prompt=messages)
        except Exception as e:
            if self.error_mode == 'raise':
                raise
            if self.error_mode == 'ignore':
                return ChatCompletionOutput(
                    model_info=self.chat_model.model_info, message=AssistantMessage(content=''), extra={'error': str(e)}
                )
            raise ValueError(f'Unknown error mode: {self.error_mode}') from e
        else:
            progress_bar.update(1)
            return output

    async def async_run(self, prompts: Prompts) -> AsyncGenerator[ChatCompletionOutput, None]:
        limiter = anyio.CapacityLimiter(self.async_capacity)
        task_created_lock = anyio.Lock()
        progress_bar = self._get_progress_bar(num_tasks=len(prompts))

        async with asyncer.create_task_group() as task_group:
            soon_values: list[asyncer.SoonValue[ChatCompletionOutput]] = []
            soon_func = task_group.soonify(self._async_run_single_task)
            for prompt in prompts:
                soon_value = soon_func(
                    prompt=prompt,
                    limiter=limiter,
                    task_created_lock=task_created_lock,
                    progress_bar=progress_bar,
                )
                soon_values.append(soon_value)
            for soon_value in soon_values:
                while not soon_value.ready:
                    await anyio.sleep(0.01)
                yield soon_value.value

        progress_bar.close()

    async def _async_run_single_task(
        self,
        prompt: Prompt,
        limiter: anyio.CapacityLimiter,
        task_created_lock: anyio.Lock,
        progress_bar: tqdm.tqdm[NoReturn],
    ) -> ChatCompletionOutput:
        messages = ensure_messages(prompt)

        async with limiter:
            try:
                async with task_created_lock:
                    sleep_time = self._calculate_sleep_time()
                    if sleep_time > 0:
                        await anyio.sleep(sleep_time)
                    self._task_created_time_list.append(int(time.time()))
                output = await self.chat_model.async_generate(messages)
            except Exception as e:
                if self.error_mode == 'raise':
                    raise
                if self.error_mode == 'ignore':
                    return ChatCompletionOutput(
                        model_info=self.chat_model.model_info, message=AssistantMessage(content=''), extra={'error': str(e)}
                    )

                raise ValueError(f'Unknown error mode: {self.error_mode}') from e
            else:
                progress_bar.update(1)
                return output

    def _calculate_sleep_time(self) -> int:
        idx = 0
        current_time = time.time()
        for i, task_created_time in enumerate(self._task_created_time_list):
            if current_time - task_created_time < self.NUM_SECONDS_PER_MINUTE:
                idx = i
                break
        self._task_created_time_list = self._task_created_time_list[idx:]

        if len(self._task_created_time_list) < self.max_requests_per_minute:
            return 0

        return max(self.NUM_SECONDS_PER_MINUTE - int(current_time - self._task_created_time_list[0]) + 1, 0)

    def _get_progress_bar(self, num_tasks: int) -> tqdm.tqdm[NoReturn]:
        use_progress_bar = (self.progress_bar_mode == 'always') or (
            self.progress_bar_mode == 'auto' and num_tasks > self.PROGRESS_BAR_THRESHOLD
        )
        return tqdm.tqdm(desc=f'{self.chat_model.__class__.__name__}', total=num_tasks, disable=not use_progress_bar)
