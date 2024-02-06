from __future__ import annotations

from typing import AsyncIterator, ClassVar, Iterator, List, Optional, Union

from pydantic import Field, PositiveInt
from typing_extensions import Annotated, Unpack, override

from generate.chat_completion.message import Prompt
from generate.chat_completion.model_output import ChatCompletionOutput, ChatCompletionStreamOutput
from generate.chat_completion.models.openai import OpenAIChat
from generate.http import HttpClient
from generate.model import ModelParameters, ModelParametersDict
from generate.platforms import DeepSeekSettings
from generate.types import Probability


class DeepSeekChatParameters(ModelParameters):
    temperature: Optional[Annotated[float, Field(ge=0, le=2)]] = None
    top_p: Optional[Probability] = None
    max_tokens: Optional[PositiveInt] = None
    frequency_penalty: Optional[Annotated[float, Field(ge=-2, le=2)]] = None
    presence_penalty: Optional[Annotated[float, Field(ge=-2, le=2)]] = None
    stop: Optional[Union[str, List[str]]] = None


class DeepSeekParametersDict(ModelParametersDict, total=False):
    temperature: float
    top_p: float
    max_tokens: int
    frequency_penalty: float
    presence_penalty: float
    stop: Union[str, List[str]]


class DeepSeekChat(OpenAIChat):
    model_type: ClassVar[str] = 'deepseek'

    parameters: DeepSeekChatParameters
    settings: DeepSeekSettings

    def __init__(
        self,
        model: str = 'deepseek-chat',
        parameters: DeepSeekChatParameters | None = None,
        settings: DeepSeekSettings | None = None,
        http_client: HttpClient | None = None,
    ) -> None:
        self.parameters = parameters or DeepSeekChatParameters()
        self.settings = settings or DeepSeekSettings()  # type: ignore
        self.http_client = http_client or HttpClient()
        self.model = model

    @override
    def generate(self, prompt: Prompt, **kwargs: Unpack[DeepSeekParametersDict]) -> ChatCompletionOutput:
        return super().generate(prompt, **kwargs)

    @override
    async def async_generate(self, prompt: Prompt, **kwargs: Unpack[DeepSeekParametersDict]) -> ChatCompletionOutput:
        return await super().async_generate(prompt, **kwargs)

    @override
    def stream_generate(self, prompt: Prompt, **kwargs: Unpack[DeepSeekParametersDict]) -> Iterator[ChatCompletionStreamOutput]:
        yield from super().stream_generate(prompt, **kwargs)

    @override
    async def async_stream_generate(
        self, prompt: Prompt, **kwargs: Unpack[DeepSeekParametersDict]
    ) -> AsyncIterator[ChatCompletionStreamOutput]:
        async for i in super().async_stream_generate(prompt, **kwargs):
            yield i
