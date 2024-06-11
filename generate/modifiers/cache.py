from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, AsyncIterator, TypeVar, cast

from diskcache import Cache
from typing_extensions import Self, override

from generate.chat_completion import ChatCompletionModel, ChatCompletionOutput, RemoteChatCompletionModel
from generate.chat_completion.message.core import Messages, Prompt
from generate.chat_completion.message.utils import ensure_messages
from generate.chat_completion.model_output import ChatCompletionStreamOutput

T = TypeVar('T')


def messages_to_text(messages: Messages) -> str:
    return '\n'.join([str(i) for i in messages])


def hash_text(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


class CacheChatCompletionModel(ChatCompletionModel):
    defalut_cache_dir = Path.home() / '.cache' / 'generate-chat-completion'

    def __init__(self, model: ChatCompletionModel, cache_dir: Path | str | None = None) -> None:
        self.model = cast(RemoteChatCompletionModel, model)
        cache_dir = cache_dir or self.defalut_cache_dir
        self.disk_cache = Cache(directory=cache_dir)
        self.model_type = self.model.model_type  # type: ignore

    @property
    def name(self) -> str:
        return self.model.name

    @classmethod
    def from_name(cls, name: str) -> Self:
        raise ValueError('Cache model cannot be created from name')

    @override
    def generate(self, prompt: Prompt, **kwargs: Any) -> ChatCompletionOutput:
        messages = ensure_messages(prompt)
        hash_key = hash_text(f'{self.model.model_id} {messages_to_text(messages)} {self.model.parameters.model_dump_json()}')
        if hash_key in self.disk_cache:
            return self.disk_cache[hash_key]  # type: ignore
        model_output = self.model.generate(messages, **kwargs)
        self.disk_cache[hash_key] = model_output
        return model_output

    @override
    async def async_generate(self, prompt: Prompt, **kwargs: Any) -> ChatCompletionOutput:
        messages = ensure_messages(prompt)
        hash_key = hash_text(f'{self.model.model_id} {messages_to_text(messages)} {self.model.parameters.model_dump_json()}')
        if hash_key in self.disk_cache:
            return self.disk_cache[hash_key]  # type: ignore
        model_output = await self.model.async_generate(messages, **kwargs)
        self.disk_cache[hash_key] = model_output
        return model_output

    @override
    async def async_stream_generate(self, prompt: Prompt, **kwargs: Any) -> AsyncIterator[ChatCompletionStreamOutput]:
        raise NotImplementedError('Stream generation is not supported for cache models')
