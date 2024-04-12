from __future__ import annotations

from typing import Any, AsyncIterator, Iterable, Iterator, NoReturn, TypeVar

from typing_extensions import Self, override

from generate.chat_completion import ChatCompletionModel, ChatCompletionOutput
from generate.chat_completion.message.core import Messages, Prompt
from generate.chat_completion.message.utils import ensure_messages
from generate.chat_completion.model_output import ChatCompletionStreamOutput

T = TypeVar('T')


class SessionChatCompletionModel(ChatCompletionModel):
    def __init__(self, model: ChatCompletionModel) -> None:
        self.model = model
        self.history: Messages = []

        self.model_type = self.model.model_type  # type: ignore

    @property
    def name(self) -> str:
        return self.model.name

    @classmethod
    def from_name(cls, name: str) -> Self:
        raise ValueError('Session model cannot be created from name')

    @override
    def generate(self, prompt: Prompt, **kwargs: Any) -> ChatCompletionOutput:
        messages = ensure_messages(prompt)
        self.history.extend(messages)
        model_output = self.model.generate(self.history, **kwargs)
        self.history.append(model_output.message)
        return model_output

    @override
    async def async_generate(self, prompt: Prompt, **kwargs: Any) -> ChatCompletionOutput:
        messages = ensure_messages(prompt)
        self.history.extend(messages)
        model_output = await self.model.async_generate(self.history, **kwargs)
        self.history.append(model_output.message)
        return model_output

    @override
    def stream_generate(self, prompt: Prompt, **kwargs: Any) -> Iterator[ChatCompletionStreamOutput]:
        messages = ensure_messages(prompt)
        self.history.extend(messages)
        finished = False
        for stream_output in self.model.stream_generate(self.history, **kwargs):
            if not finished:
                yield stream_output
            if stream_output.is_finish:
                finished = True
                self.history.append(stream_output.message)

    @override
    async def async_stream_generate(self, prompt: Prompt, **kwargs: Any) -> AsyncIterator[ChatCompletionStreamOutput]:
        messages = ensure_messages(prompt)
        self.history.extend(messages)
        finished = False
        async for stream_output in self.model.async_stream_generate(self.history, **kwargs):
            if not finished:
                yield stream_output
            if stream_output.is_finish:
                finished = True
                self.history.append(stream_output.message)

    async def async_batch_generate(self, prompts: Iterable[Prompt], **kwargs: Any) -> NoReturn:
        raise RuntimeError('Async Batch generation is not supported for session model')

    def reset(self) -> None:
        self.history.clear()
