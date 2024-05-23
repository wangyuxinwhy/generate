from __future__ import annotations

from typing import AsyncIterator, ClassVar, Iterator, List, Literal, Optional

from pydantic import Field
from typing_extensions import Annotated, NotRequired, TypedDict, Unpack, override

from generate.chat_completion.message import (
    Prompt,
)
from generate.chat_completion.model_output import ChatCompletionOutput, ChatCompletionStreamOutput
from generate.chat_completion.models.openai_like import OpenAILikeChat, SupportOpenAIToolCall
from generate.chat_completion.tool import FunctionJsonSchema
from generate.http import (
    HttpClient,
)
from generate.model import ModelParameters, RemoteModelParametersDict
from generate.platforms.baichuan import BaichuanSettings
from generate.types import Probability, Temperature


class BaichuanResponseFormat(TypedDict):
    type: Literal['json_object']


class BaichuanRetrieval(TypedDict):
    kb_ids: List[str]
    answer_model: NotRequired[str]


class BaichuanWebSearch(TypedDict):
    enable: NotRequired[bool]
    search_mode: NotRequired[str]


class BaichuanTool(TypedDict):
    type: Literal['retrieval', 'web_search', 'function']
    retrieval: NotRequired[BaichuanRetrieval]
    web_search: NotRequired[BaichuanWebSearch]
    function: NotRequired[FunctionJsonSchema]


class BaichuanChatParameters(ModelParameters):
    temperature: Optional[Temperature] = None
    top_k: Optional[Annotated[int, Field(ge=0)]] = None
    top_p: Optional[Probability] = None
    max_tokens: Optional[Annotated[int, Field(ge=0)]] = None
    response_format: Optional[BaichuanResponseFormat] = None
    tools: Optional[List[BaichuanTool]] = None
    tool_choice: Optional[str] = None


class BaichuanChatParametersDict(RemoteModelParametersDict, total=False):
    temperature: Optional[Temperature]
    top_k: Optional[int]
    top_p: Optional[Probability]
    max_tokens: Optional[int]
    response_format: Optional[BaichuanResponseFormat]
    tools: Optional[List[BaichuanTool]]
    tool_choice: Optional[str]


class BaichuanChat(OpenAILikeChat, SupportOpenAIToolCall):
    model_type: ClassVar[str] = 'baichuan'
    available_models: ClassVar[List[str]] = [
        'Baichuan2-Turbo',
        'Baichuan2-Turbo-192k',
        'Baichuan3-Turbo',
        'Baichuan3-Turbo-128k',
        'Baichuan4',
    ]

    parameters: BaichuanChatParameters
    settings: BaichuanSettings

    def __init__(
        self,
        model: str = 'Baichuan3-Turbo',
        parameters: BaichuanChatParameters | None = None,
        settings: BaichuanSettings | None = None,
        http_client: HttpClient | None = None,
    ) -> None:
        parameters = parameters or BaichuanChatParameters()
        settings = settings or BaichuanSettings()  # type: ignore
        http_client = http_client or HttpClient()
        model = model
        super().__init__(model=model, parameters=parameters, settings=settings, http_client=http_client)

    @override
    def generate(self, prompt: Prompt, **kwargs: Unpack[BaichuanChatParametersDict]) -> ChatCompletionOutput:
        return super().generate(prompt, **kwargs)

    @override
    async def async_generate(self, prompt: Prompt, **kwargs: Unpack[BaichuanChatParametersDict]) -> ChatCompletionOutput:
        return await super().async_generate(prompt, **kwargs)

    @override
    def stream_generate(
        self, prompt: Prompt, **kwargs: Unpack[BaichuanChatParametersDict]
    ) -> Iterator[ChatCompletionStreamOutput]:
        yield from super().stream_generate(prompt, **kwargs)

    @override
    async def async_stream_generate(
        self, prompt: Prompt, **kwargs: Unpack[BaichuanChatParametersDict]
    ) -> AsyncIterator[ChatCompletionStreamOutput]:
        async for stream_output in super().async_stream_generate(prompt, **kwargs):
            yield stream_output
