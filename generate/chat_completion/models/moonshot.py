from __future__ import annotations

from typing import AsyncIterator, ClassVar, Iterator, Optional

from pydantic import PositiveInt
from typing_extensions import Unpack, override

from generate.chat_completion.message import Prompt
from generate.chat_completion.model_output import ChatCompletionOutput, ChatCompletionStreamOutput
from generate.chat_completion.models.openai import OpenAIChat
from generate.http import HttpClient
from generate.model import ModelParameters, ModelParametersDict
from generate.platforms import MoonshotSettings
from generate.types import Probability, Temperature


class MoonshotParameters(ModelParameters):
    temperature: Optional[Temperature] = None
    top_p: Optional[Probability] = None
    max_tokens: Optional[PositiveInt] = None


class MoonshotParametersDict(ModelParametersDict, total=False):
    temperature: Temperature
    top_p: Probability
    max_tokens: PositiveInt


class MoonshotChat(OpenAIChat):
    model_type: ClassVar[str] = 'moonshot'

    parameters: MoonshotParameters
    settings: MoonshotSettings

    def __init__(
        self,
        model: str = 'moonshot-v1-8k',
        parameters: MoonshotParameters | None = None,
        settings: MoonshotSettings | None = None,
        http_client: HttpClient | None = None,
    ) -> None:
        self.parameters = parameters or MoonshotParameters()
        self.settings = settings or MoonshotSettings()  # type: ignore
        self.http_client = http_client or HttpClient()
        self.model = model

    @override
    def generate(self, prompt: Prompt, **kwargs: Unpack[MoonshotParametersDict]) -> ChatCompletionOutput:
        return super().generate(prompt, **kwargs)

    @override
    async def async_generate(self, prompt: Prompt, **kwargs: Unpack[MoonshotParametersDict]) -> ChatCompletionOutput:
        return await super().async_generate(prompt, **kwargs)

    @override
    def stream_generate(self, prompt: Prompt, **kwargs: Unpack[MoonshotParametersDict]) -> Iterator[ChatCompletionStreamOutput]:
        yield from super().stream_generate(prompt, **kwargs)

    @override
    async def async_stream_generate(
        self, prompt: Prompt, **kwargs: Unpack[MoonshotParametersDict]
    ) -> AsyncIterator[ChatCompletionStreamOutput]:
        async for i in super().async_stream_generate(prompt, **kwargs):
            yield i
