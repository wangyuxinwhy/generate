from __future__ import annotations

from typing import AsyncIterator, ClassVar, Iterator, List, Optional, Union

from pydantic import Field, PositiveInt
from typing_extensions import Annotated, Unpack, override

from generate.chat_completion.cost_caculator import CostCalculator, GeneralCostCalculator
from generate.chat_completion.message import Prompt
from generate.chat_completion.model_output import ChatCompletionOutput, ChatCompletionStreamOutput
from generate.chat_completion.models.openai_like import OpenAILikeChat, OpenAIMessageConverter
from generate.http import HttpClient
from generate.model import ModelParameters, RemoteModelParametersDict
from generate.platforms import DeepSeekSettings
from generate.types import ModelPrice, Probability

DeepSeekModelPrice: ModelPrice = {
    'deepseek-chat': (1, 2),
    'deepseek-coder': (1, 2),
}


class DeepSeekChatParameters(ModelParameters):
    temperature: Optional[Annotated[float, Field(ge=0, le=2)]] = None
    top_p: Optional[Probability] = None
    max_tokens: Optional[PositiveInt] = None
    frequency_penalty: Optional[Annotated[float, Field(ge=-2, le=2)]] = None
    presence_penalty: Optional[Annotated[float, Field(ge=-2, le=2)]] = None
    stop: Optional[Union[str, List[str]]] = None
    logprobs: Optional[bool] = None
    top_logprobs: Optional[Annotated[int, Field(ge=0, le=20)]] = None


class DeepSeekParametersDict(RemoteModelParametersDict, total=False):
    temperature: Optional[float]
    top_p: Optional[Probability]
    max_tokens: Optional[PositiveInt]
    frequency_penalty: Optional[float]
    presence_penalty: Optional[float]
    stop: Optional[Union[str, List[str]]]
    logprobs: Optional[bool]
    top_logprobs: Optional[int]


class DeepSeekChat(OpenAILikeChat):
    model_type: ClassVar[str] = 'deepseek'
    available_models: ClassVar[List[str]] = ['deepseek-chat', 'deepseek-coder']

    parameters: DeepSeekChatParameters
    settings: DeepSeekSettings
    message_converter: OpenAIMessageConverter

    def __init__(
        self,
        model: str = 'deepseek-chat',
        parameters: DeepSeekChatParameters | None = None,
        settings: DeepSeekSettings | None = None,
        http_client: HttpClient | None = None,
        message_converter: OpenAIMessageConverter | None = None,
        cost_calculator: CostCalculator | None = None,
    ) -> None:
        parameters = parameters or DeepSeekChatParameters()
        settings = settings or DeepSeekSettings()  # type: ignore
        cost_calculator = cost_calculator or GeneralCostCalculator(DeepSeekModelPrice)
        super().__init__(
            model=model,
            parameters=parameters,
            settings=settings,
            http_client=http_client,
            message_converter=message_converter,
            cost_calculator=cost_calculator,
        )

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
        async for stream_output in super().async_stream_generate(prompt, **kwargs):
            yield stream_output
