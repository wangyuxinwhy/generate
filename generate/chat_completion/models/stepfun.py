from __future__ import annotations

from typing import AsyncIterator, ClassVar, Iterator, List, Optional

from pydantic import Field, PositiveInt
from typing_extensions import Annotated, Unpack, override

from generate.chat_completion.message import Prompt
from generate.chat_completion.model_output import ChatCompletionOutput, ChatCompletionStreamOutput
from generate.chat_completion.models.openai_like import OpenAILikeChat
from generate.http import HttpClient
from generate.model import ModelParameters, RemoteModelParametersDict
from generate.platforms import StepFunSettings
from generate.types import Probability

Temperature = Annotated[float, Field(ge=0, le=2)]


class StepFunChatParameters(ModelParameters):
    temperature: Optional[Temperature] = None
    top_p: Optional[Probability] = None
    max_tokens: Optional[PositiveInt] = None
    presence_penalty: Optional[Annotated[float, Field(ge=-2, le=2)]] = None
    frequency_penalty: Optional[Annotated[float, Field(ge=-2, le=2)]] = None


class StepFunParametersDict(RemoteModelParametersDict, total=False):
    temperature: Optional[Temperature]
    top_p: Optional[Probability]
    max_tokens: Optional[PositiveInt]
    presence_penalty: Optional[float]
    frequency_penalty: Optional[float]


class StepFunChat(OpenAILikeChat):
    model_type: ClassVar[str] = 'stepfun'
    available_models: ClassVar[List[str]] = ['step-1-32k', 'step-1v-32k', 'step-1-200k']

    parameters: StepFunChatParameters
    settings: StepFunSettings

    def __init__(
        self,
        model: str = 'step-1-32k',
        parameters: StepFunChatParameters | None = None,
        settings: StepFunSettings | None = None,
        http_client: HttpClient | None = None,
    ) -> None:
        parameters = parameters or StepFunChatParameters()
        settings = settings or StepFunSettings()  # type: ignore
        http_client = http_client or HttpClient()
        model = model
        super().__init__(model=model, parameters=parameters, settings=settings, http_client=http_client)

    @override
    def generate(self, prompt: Prompt, **kwargs: Unpack[StepFunParametersDict]) -> ChatCompletionOutput:
        return super().generate(prompt, **kwargs)

    @override
    async def async_generate(self, prompt: Prompt, **kwargs: Unpack[StepFunParametersDict]) -> ChatCompletionOutput:
        return await super().async_generate(prompt, **kwargs)

    @override
    def stream_generate(self, prompt: Prompt, **kwargs: Unpack[StepFunParametersDict]) -> Iterator[ChatCompletionStreamOutput]:
        yield from super().stream_generate(prompt, **kwargs)

    @override
    async def async_stream_generate(
        self, prompt: Prompt, **kwargs: Unpack[StepFunParametersDict]
    ) -> AsyncIterator[ChatCompletionStreamOutput]:
        async for stream_output in super().async_stream_generate(prompt, **kwargs):
            yield stream_output
