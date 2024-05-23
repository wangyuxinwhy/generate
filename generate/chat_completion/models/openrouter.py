from __future__ import annotations

from typing import Any, AsyncIterator, ClassVar, Dict, Iterator, List, Literal, Optional, Union

from pydantic import BaseModel, Field, PositiveInt
from typing_extensions import Annotated, Unpack, override

from generate.chat_completion.message import Prompt
from generate.chat_completion.message.core import Messages
from generate.chat_completion.model_output import ChatCompletionOutput, ChatCompletionStreamOutput
from generate.chat_completion.models.openai_like import (
    OpenAILikeChat,
    OpenAIResponseFormat,
    OpenAITool,
    OpenAIToolChoice,
    SupportOpenAIToolCall,
)
from generate.http import HttpClient, HttpxPostKwargs
from generate.model import ModelParameters, RemoteModelParametersDict
from generate.platforms import OpenRouterSettings
from generate.types import Probability, Temperature


class ProviderParameters(BaseModel):
    allow_fallbacks: bool = True
    require_parameters: bool = False
    data_collection: str = 'allow'
    order: Optional[List[str]] = None


class OpenRouterChatParameters(ModelParameters):
    temperature: Optional[Temperature] = None
    top_p: Optional[Probability] = None
    top_k: Optional[PositiveInt] = None
    max_tokens: Optional[PositiveInt] = None
    stop: Union[str, List[str], None] = None
    presence_penalty: Optional[Annotated[float, Field(ge=-2, le=2)]] = None
    frequency_penalty: Optional[Annotated[float, Field(ge=-2, le=2)]] = None
    repetition_penalty: Optional[Annotated[float, Field(ge=0.0, le=2.0)]] = None
    logit_bias: Optional[Dict[int, Annotated[int, Field(ge=-100, le=100)]]] = None
    user: Optional[str] = None
    response_format: Optional[OpenAIResponseFormat] = None
    seed: Optional[int] = None
    tools: Optional[List[OpenAITool]] = None
    tool_choice: Union[Literal['auto', 'none'], OpenAIToolChoice, None] = None
    route: Optional[str] = None
    transforms: Optional[List[str]] = None
    provider: Optional[ProviderParameters] = None


class OpenRouterParametersDict(RemoteModelParametersDict, total=False):
    temperature: Optional[Temperature]
    top_p: Optional[Probability]
    top_k: Optional[PositiveInt]
    max_tokens: Optional[PositiveInt]
    stop: Union[str, List[str], None]
    presence_penalty: Optional[float]
    frequency_penalty: Optional[float]
    repetition_penalty: Optional[float]
    logit_bias: Optional[Dict[int, int]]
    user: Optional[str]
    response_format: Optional[OpenAIResponseFormat]
    seed: Optional[int]
    tools: Optional[List[OpenAITool]]
    tool_choice: Union[Literal['auto', 'none'], OpenAIToolChoice, None]
    route: Optional[str]
    transforms: Optional[List[str]]
    provider: ProviderParameters


class OpenRouterChat(OpenAILikeChat, SupportOpenAIToolCall):
    model_type: ClassVar[str] = 'openrouter'
    available_models: ClassVar[List[str]] = ['auto']

    parameters: OpenRouterChatParameters
    settings: OpenRouterSettings

    def __init__(
        self,
        model: str | list[str] = 'auto',
        parameters: OpenRouterChatParameters | None = None,
        settings: OpenRouterSettings | None = None,
        http_client: HttpClient | None = None,
        app_name: str | None = None,
        site_url: str | None = None,
    ) -> None:
        parameters = parameters or OpenRouterChatParameters()
        settings = settings or OpenRouterSettings()  # type: ignore
        http_client = http_client or HttpClient()
        if isinstance(model, list):
            self.models = model
            model = '-'.join(model[:3])
            if len(model) > 3:
                model += '-etc'
        else:
            self.models = None
        super().__init__(model=model, parameters=parameters, settings=settings, http_client=http_client)
        self.app_name = app_name
        self.site_url = site_url

    @override
    def generate(self, prompt: Prompt, **kwargs: Unpack[OpenRouterParametersDict]) -> ChatCompletionOutput:
        return super().generate(prompt, **kwargs)  # type: ignore

    @override
    async def async_generate(self, prompt: Prompt, **kwargs: Unpack[OpenRouterParametersDict]) -> ChatCompletionOutput:
        return await super().async_generate(prompt, **kwargs)  # type: ignore

    @override
    def stream_generate(
        self, prompt: Prompt, **kwargs: Unpack[OpenRouterParametersDict]
    ) -> Iterator[ChatCompletionStreamOutput]:
        yield from super().stream_generate(prompt, **kwargs)  # type: ignore

    @override
    async def async_stream_generate(
        self, prompt: Prompt, **kwargs: Unpack[OpenRouterParametersDict]
    ) -> AsyncIterator[ChatCompletionStreamOutput]:
        async for stream_output in super().async_stream_generate(prompt, **kwargs):  # type: ignore
            yield stream_output

    @override
    def _get_request_parameters(self, messages: Messages, stream: bool = False, **kwargs: Any) -> HttpxPostKwargs:
        request_parameters: HttpxPostKwargs = super()._get_request_parameters(messages, stream=stream, **kwargs)
        if self.app_name:
            request_parameters['headers']['X-Title'] = self.app_name
        if self.site_url:
            request_parameters['headers']['HTTP-Referer'] = self.site_url
        if self.models:
            request_parameters['json']['models'] = self.models
            request_parameters['json'].pop('model')
        return request_parameters
