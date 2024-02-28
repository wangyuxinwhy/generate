from __future__ import annotations

from typing import Any, AsyncIterator, Dict, Iterator, Protocol, Sequence, TypedDict

from pydantic import BaseModel, ConfigDict
from typing_extensions import Self, override

from generate.chat_completion import ChatCompletionModel, ChatCompletionOutput
from generate.chat_completion.message.core import Prompt
from generate.chat_completion.model_output import ChatCompletionStreamOutput


class BeforeGenerateContext(BaseModel):
    model_config = ConfigDict(protected_namespaces=(), arbitrary_types_allowed=True)

    prompt: Prompt
    model: ChatCompletionModel
    generate_kwargs: Dict[str, Any]


class AfterGenerateContext(BaseModel):
    model_config = ConfigDict(protected_namespaces=(), arbitrary_types_allowed=True)

    prompt: Prompt
    model: ChatCompletionModel
    model_output: ChatCompletionOutput
    generate_kwargs: Dict[str, Any]


class BeforeGenerateHook(Protocol):
    def __call__(self, context: BeforeGenerateContext) -> BeforeGenerateContext:
        ...


class AfterGenerateHook(Protocol):
    def __call__(self, context: AfterGenerateContext) -> AfterGenerateContext:
        ...


class HookModelKwargs(TypedDict, total=False):
    before_generate_hooks: Sequence[BeforeGenerateHook]
    after_generate_hooks: Sequence[AfterGenerateHook]


class HookChatCompletionModel(ChatCompletionModel):
    def __init__(
        self,
        model: ChatCompletionModel,
        before_generate_hooks: Sequence[BeforeGenerateHook] | None = None,
        after_generate_hooks: Sequence[AfterGenerateHook] | None = None,
    ) -> None:
        self.model = model
        self.before_generate_hooks = before_generate_hooks or []
        self.after_generate_hooks = after_generate_hooks or []

        self.model_type = self.model.model_type  # type: ignore

    @property
    def name(self) -> str:
        return self.model.name

    @classmethod
    def from_name(cls, name: str) -> Self:
        raise ValueError('Session model cannot be created from name')

    @override
    def generate(self, prompt: Prompt, **kwargs: Any) -> ChatCompletionOutput:
        before_context = BeforeGenerateContext(prompt=prompt, model=self.model, generate_kwargs=kwargs)
        for hook in self.before_generate_hooks:
            before_context = hook(before_context)
        model_output = self.model.generate(before_context.prompt, **kwargs)
        after_context = AfterGenerateContext(prompt=prompt, model=self.model, model_output=model_output, generate_kwargs=kwargs)
        for hook in self.after_generate_hooks:
            after_context = hook(after_context)
        return after_context.model_output

    @override
    async def async_generate(self, prompt: Prompt, **kwargs: Any) -> ChatCompletionOutput:
        before_context = BeforeGenerateContext(prompt=prompt, model=self.model, generate_kwargs=kwargs)
        for hook in self.before_generate_hooks:
            before_context = hook(before_context)
        model_output = await self.model.async_generate(before_context.prompt, **kwargs)
        after_context = AfterGenerateContext(prompt=prompt, model=self.model, model_output=model_output, generate_kwargs=kwargs)
        for hook in self.after_generate_hooks:
            after_context = hook(after_context)
        return after_context.model_output

    @override
    def stream_generate(self, prompt: Prompt, **kwargs: Any) -> Iterator[ChatCompletionStreamOutput]:
        before_context = BeforeGenerateContext(prompt=prompt, model=self.model, generate_kwargs=kwargs)
        for hook in self.before_generate_hooks:
            before_context = hook(before_context)
        for stream_output in self.model.stream_generate(before_context.prompt, **kwargs):
            after_context = AfterGenerateContext(
                prompt=prompt, model=self.model, model_output=stream_output, generate_kwargs=kwargs
            )
            for hook in self.after_generate_hooks:
                after_context = hook(after_context)
            assert isinstance(after_context.model_output, ChatCompletionStreamOutput)
            yield after_context.model_output
            if after_context.model_output.is_finish:
                break

    @override
    async def async_stream_generate(self, prompt: Prompt, **kwargs: Any) -> AsyncIterator[ChatCompletionStreamOutput]:
        before_context = BeforeGenerateContext(prompt=prompt, model=self.model, generate_kwargs=kwargs)
        for hook in self.before_generate_hooks:
            before_context = hook(before_context)
        async for stream_output in self.model.async_stream_generate(before_context.prompt, **kwargs):
            after_context = AfterGenerateContext(
                prompt=prompt, model=self.model, model_output=stream_output, generate_kwargs=kwargs
            )
            for hook in self.after_generate_hooks:
                after_context = hook(after_context)
            assert isinstance(after_context.model_output, ChatCompletionStreamOutput)
            yield after_context.model_output
            if after_context.model_output.is_finish:
                break
