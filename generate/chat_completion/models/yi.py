from __future__ import annotations

from typing import AsyncIterator, ClassVar, Iterator, Optional

from pydantic import Field, PositiveInt
from typing_extensions import Annotated, Unpack, override

from generate.chat_completion.message import Prompt
from generate.chat_completion.model_output import ChatCompletionOutput, ChatCompletionStreamOutput
from generate.chat_completion.models.openai import OpenAIChat
from generate.http import HttpClient
from generate.model import ModelParameters, ModelParametersDict
from generate.platforms import YiSettings


class YiChatParameters(ModelParameters):
    temperature: Optional[Annotated[float, Field(ge=0, lt=2)]] = None
    max_tokens: Optional[PositiveInt] = None


class YiParametersDict(ModelParametersDict, total=False):
    temperature: float
    max_tokens: int


class YiChat(OpenAIChat):
    model_type: ClassVar[str] = 'yi'

    parameters: YiChatParameters
    settings: YiSettings

    def __init__(
        self,
        model: str = 'yi-34b-chat',
        parameters: YiChatParameters | None = None,
        settings: YiSettings | None = None,
        http_client: HttpClient | None = None,
    ) -> None:
        self.parameters = parameters or YiChatParameters()
        self.settings = settings or YiSettings()  # type: ignore
        self.http_client = http_client or HttpClient()
        self.model = model

    @override
    def generate(self, prompt: Prompt, **kwargs: Unpack[YiParametersDict]) -> ChatCompletionOutput:
        return super().generate(prompt, **kwargs)

    @override
    async def async_generate(self, prompt: Prompt, **kwargs: Unpack[YiParametersDict]) -> ChatCompletionOutput:
        return await super().async_generate(prompt, **kwargs)

    @override
    def stream_generate(self, prompt: Prompt, **kwargs: Unpack[YiParametersDict]) -> Iterator[ChatCompletionStreamOutput]:
        yield from super().stream_generate(prompt, **kwargs)

    @override
    async def async_stream_generate(
        self, prompt: Prompt, **kwargs: Unpack[YiParametersDict]
    ) -> AsyncIterator[ChatCompletionStreamOutput]:
        async for i in super().async_stream_generate(prompt, **kwargs):
            yield i
