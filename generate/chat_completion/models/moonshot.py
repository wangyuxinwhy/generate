from __future__ import annotations

from typing import AsyncIterator, ClassVar, Iterator, List, Optional

from pydantic import PositiveInt
from typing_extensions import Unpack, override

from generate.chat_completion.message import Prompt
from generate.chat_completion.model_output import ChatCompletionOutput, ChatCompletionStreamOutput
from generate.chat_completion.models.openai_like import OpenAILikeChat
from generate.http import HttpClient
from generate.model import ModelParameters, RemoteModelParametersDict
from generate.platforms import MoonshotSettings
from generate.types import Probability, Temperature


class MoonshotChatParameters(ModelParameters):
    temperature: Optional[Temperature] = None
    top_p: Optional[Probability] = None
    max_tokens: Optional[PositiveInt] = None


class MoonshotParametersDict(RemoteModelParametersDict, total=False):
    temperature: Temperature
    top_p: Probability
    max_tokens: PositiveInt


class MoonshotChat(OpenAILikeChat):
    model_type: ClassVar[str] = 'moonshot'
    available_models: ClassVar[List[str]] = ['moonshot-v1-8k', 'moonshot-v1-32k', 'moonshot-v1-128k']

    parameters: MoonshotChatParameters
    settings: MoonshotSettings

    def __init__(
        self,
        model: str = 'moonshot-v1-8k',
        parameters: MoonshotChatParameters | None = None,
        settings: MoonshotSettings | None = None,
        http_client: HttpClient | None = None,
    ) -> None:
        parameters = parameters or MoonshotChatParameters()
        settings = settings or MoonshotSettings()  # type: ignore
        http_client = http_client or HttpClient()
        model = model
        super().__init__(model=model, parameters=parameters, settings=settings, http_client=http_client)

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
        async for stream_output in super().async_stream_generate(prompt, **kwargs):
            yield stream_output
