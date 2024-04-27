from __future__ import annotations

from typing import AsyncIterator, ClassVar, Iterator, List, Optional

from pydantic import Field, PositiveInt
from typing_extensions import Annotated, Unpack, override

from generate.chat_completion.cost_caculator import CostCalculator, GeneralCostCalculator
from generate.chat_completion.message import Prompt
from generate.chat_completion.model_output import ChatCompletionOutput, ChatCompletionStreamOutput
from generate.chat_completion.models.openai_like import OpenAILikeChat, OpenAIMessageConverter
from generate.http import HttpClient
from generate.model import ModelParameters, RemoteModelParametersDict
from generate.platforms import YiSettings
from generate.types import ModelPrice

YiModelPrice: ModelPrice = {
    'yi-34b-chat-200k': (12.0, 12.0),
    'yi-34b-chat': (2.5, 2.5),
    'yi-vl-plus': (6, 6),
}


class YiChatParameters(ModelParameters):
    temperature: Optional[Annotated[float, Field(ge=0, lt=2)]] = None
    max_tokens: Optional[PositiveInt] = None
    top_p: Optional[Annotated[float, Field(ge=0, lt=1)]] = None


class YiParametersDict(RemoteModelParametersDict, total=False):
    temperature: Optional[float]
    max_tokens: Optional[int]
    top_p: Optional[float]


class YiChat(OpenAILikeChat):
    model_type: ClassVar[str] = 'yi'
    available_models: ClassVar[List[str]] = ['yi-34b-chat-0205', 'yi-34b-chat-200k', 'yi-vl-plus']

    parameters: YiChatParameters
    settings: YiSettings
    message_converter: OpenAIMessageConverter

    def __init__(
        self,
        model: str = 'yi-34b-chat-0205',
        parameters: YiChatParameters | None = None,
        settings: YiSettings | None = None,
        http_client: HttpClient | None = None,
        message_converter: OpenAIMessageConverter | None = None,
        cost_calculator: CostCalculator | None = None,
    ) -> None:
        parameters = parameters or YiChatParameters()
        settings = settings or YiSettings()  # type: ignore
        cost_calculator = cost_calculator or GeneralCostCalculator(YiModelPrice)
        super().__init__(
            model=model,
            parameters=parameters,
            settings=settings,
            http_client=http_client,
            message_converter=message_converter,
            cost_calculator=cost_calculator,
        )

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
        async for stream_output in super().async_stream_generate(prompt, **kwargs):
            yield stream_output
