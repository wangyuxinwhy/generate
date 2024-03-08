from __future__ import annotations

from typing import AsyncIterator, ClassVar, Dict, Iterator, List, Literal, Optional, Union

from pydantic import Field, PositiveInt
from typing_extensions import Annotated, Unpack, override

from generate.chat_completion.message import Prompt
from generate.chat_completion.model_output import ChatCompletionOutput, ChatCompletionStreamOutput
from generate.chat_completion.models.openai_like import (
    FunctionCallName,
    OpenAILikeChat,
    OpenAIResponseFormat,
    OpenAITool,
    OpenAIToolChoice,
    convert_to_openai_tool,
)
from generate.chat_completion.tool import FunctionJsonSchema, Tool, ToolCallMixin
from generate.http import (
    HttpClient,
)
from generate.model import ModelParameters, RemoteModelParametersDict
from generate.platforms.openai import OpenAISettings
from generate.types import OrIterable, Probability, Temperature
from generate.utils import ensure_iterable


class OpenAIChatParameters(ModelParameters):
    temperature: Optional[Temperature] = None
    top_p: Optional[Probability] = None
    max_tokens: Optional[PositiveInt] = None
    functions: Optional[List[FunctionJsonSchema]] = None
    function_call: Union[Literal['auto'], FunctionCallName, None] = None
    stop: Union[str, List[str], None] = None
    presence_penalty: Optional[Annotated[float, Field(ge=-2, le=2)]] = None
    frequency_penalty: Optional[Annotated[float, Field(ge=-2, le=2)]] = None
    logit_bias: Optional[Dict[int, Annotated[int, Field(ge=-100, le=100)]]] = None
    user: Optional[str] = None
    response_format: Optional[OpenAIResponseFormat] = None
    seed: Optional[int] = None
    tools: Optional[List[OpenAITool]] = None
    tool_choice: Union[Literal['auto'], OpenAIToolChoice, None] = None


class OpenAIChatParametersDict(RemoteModelParametersDict, total=False):
    temperature: Optional[Temperature]
    top_p: Optional[Probability]
    max_tokens: Optional[PositiveInt]
    functions: Optional[List[FunctionJsonSchema]]
    function_call: Union[Literal['auto'], FunctionCallName, None]
    stop: Union[str, List[str], None]
    presence_penalty: Optional[float]
    frequency_penalty: Optional[float]
    logit_bias: Optional[Dict[int, int]]
    user: Optional[str]
    response_format: Optional[OpenAIResponseFormat]
    seed: Optional[int]
    tools: Optional[List[OpenAITool]]
    tool_choice: Union[Literal['auto'], OpenAIToolChoice, None]


class OpenAIChat(OpenAILikeChat, ToolCallMixin):
    model_type: ClassVar[str] = 'openai'
    available_models: ClassVar[List[str]] = [
        'gpt-4-turbo-preview',
        'gpt-3.5-turbo',
        'gpt-4-vision-preview',
    ]

    parameters: OpenAIChatParameters
    settings: OpenAISettings

    def __init__(
        self,
        model: str = 'gpt-3.5-turbo',
        parameters: OpenAIChatParameters | None = None,
        settings: OpenAISettings | None = None,
        http_client: HttpClient | None = None,
    ) -> None:
        parameters = parameters or OpenAIChatParameters()
        settings = settings or OpenAISettings()  # type: ignore
        http_client = http_client or HttpClient()
        super().__init__(model=model, parameters=parameters, settings=settings, http_client=http_client)

    @override
    def generate(self, prompt: Prompt, **kwargs: Unpack[OpenAIChatParametersDict]) -> ChatCompletionOutput:
        return super().generate(prompt, **kwargs)

    @override
    async def async_generate(self, prompt: Prompt, **kwargs: Unpack[OpenAIChatParametersDict]) -> ChatCompletionOutput:
        return await super().async_generate(prompt, **kwargs)

    @override
    def stream_generate(
        self, prompt: Prompt, **kwargs: Unpack[OpenAIChatParametersDict]
    ) -> Iterator[ChatCompletionStreamOutput]:
        yield from super().stream_generate(prompt, **kwargs)

    @override
    async def async_stream_generate(
        self, prompt: Prompt, **kwargs: Unpack[OpenAIChatParametersDict]
    ) -> AsyncIterator[ChatCompletionStreamOutput]:
        async for stream_output in super().async_stream_generate(prompt, **kwargs):
            yield stream_output

    @override
    def add_tools(self, tools: OrIterable[Tool]) -> None:
        new_tools = [convert_to_openai_tool(tool) for tool in ensure_iterable(tools)]
        if self.parameters.tools is None:
            self.parameters.tools = new_tools
        else:
            self.parameters.tools.extend(new_tools)
