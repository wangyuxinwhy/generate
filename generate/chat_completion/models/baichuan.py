from __future__ import annotations

import json
from datetime import datetime
from typing import AsyncIterator, ClassVar, Iterator, List, Literal, Optional

from pydantic import Field
from typing_extensions import Annotated, TypedDict, Unpack, override

from generate.chat_completion.base import RemoteChatCompletionModel
from generate.chat_completion.message import (
    AssistantMessage,
    Message,
    Messages,
    MessageTypeError,
    Prompt,
    SystemMessage,
    UserMessage,
    ensure_messages,
)
from generate.chat_completion.model_output import ChatCompletionOutput, ChatCompletionStreamOutput
from generate.chat_completion.stream_manager import StreamManager
from generate.http import (
    HttpClient,
    HttpxPostKwargs,
    ResponseValue,
    UnexpectedResponseError,
)
from generate.model import ModelParameters, RemoteModelParametersDict
from generate.platforms.baichuan import BaichuanSettings
from generate.types import Probability, Temperature


class BaichuanMessage(TypedDict):
    role: Literal['user', 'assistant']
    content: str


class BaichuanChatParameters(ModelParameters):
    temperature: Optional[Temperature] = None
    top_k: Optional[Annotated[int, Field(ge=0)]] = None
    top_p: Optional[Probability] = None
    search: Optional[bool] = Field(default=None, alias='with_search_enhance')


class BaichuanChatParametersDict(RemoteModelParametersDict, total=False):
    temperature: Optional[Temperature]
    top_k: Optional[int]
    top_p: Optional[Probability]
    search: Optional[bool]


class BaichuanChat(RemoteChatCompletionModel):
    model_type: ClassVar[str] = 'baichuan'
    available_models: ClassVar[List[str]] = ['Baichuan2-Turbo', 'Baichuan2-53B', 'Baichuan2-Turbo-192k']

    parameters: BaichuanChatParameters
    settings: BaichuanSettings

    def __init__(
        self,
        model: str = 'Baichuan2-Turbo',
        parameters: BaichuanChatParameters | None = None,
        settings: BaichuanSettings | None = None,
        http_client: HttpClient | None = None,
    ) -> None:
        parameters = parameters or BaichuanChatParameters()
        settings = settings or BaichuanSettings()  # type: ignore
        http_client = http_client or HttpClient()
        super().__init__(model, parameters=parameters, settings=settings, http_client=http_client)

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
        async for output in super().async_stream_generate(prompt, **kwargs):
            yield output

    @override
    def _get_request_parameters(
        self, prompt: Prompt, stream: bool = False, **kwargs: Unpack[BaichuanChatParametersDict]
    ) -> HttpxPostKwargs:
        messages = ensure_messages(prompt)
        parameters = self.parameters.clone_with_changes(**kwargs)
        baichuan_messages: list[BaichuanMessage] = self._convert_messages(messages)
        data = {
            'model': self.model,
            'messages': baichuan_messages,
        }
        parameters_dict = parameters.custom_model_dump()
        data.update(parameters_dict)
        if stream:
            data['stream'] = True
        headers = {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer ' + self.settings.api_key.get_secret_value(),
        }
        return {
            'url': self.settings.api_base + '/chat/completions',
            'headers': headers,
            'json': data,
        }

    @staticmethod
    def _convert_message(message: Message) -> BaichuanMessage:
        if isinstance(message, UserMessage):
            return {
                'role': 'user',
                'content': message.content,
            }

        if isinstance(message, AssistantMessage):
            return {
                'role': 'assistant',
                'content': message.content,
            }
        raise MessageTypeError(message, (UserMessage, AssistantMessage))

    def _convert_messages(self, messages: Messages) -> list[BaichuanMessage]:
        if isinstance(system_message := messages[0], SystemMessage):
            prepend_messages = [UserMessage(content=system_message.content), AssistantMessage(content='好的')]
            messages = prepend_messages + messages[1:]
        return [self._convert_message(message) for message in messages]

    @override
    def _process_reponse(self, response: ResponseValue) -> ChatCompletionOutput:
        try:
            text = response['choices'][0]['message']['content']
            finish_reason = response['choices'][0]['finish_reason'] or None
            usage = response.get('usage')
            extra = {'id': response['id']}
            if usage is not None:
                cost = self._calculate_cost(usage['total_tokens'])
                extra['usage'] = usage
            else:
                cost = None
            return ChatCompletionOutput(
                model_info=self.model_info,
                message=AssistantMessage(content=text),
                finish_reason=finish_reason,
                cost=cost,
                extra=extra,
            )
        except (KeyError, IndexError) as e:
            raise UnexpectedResponseError(response) from e

    @override
    def _process_stream_line(self, line: str, stream_manager: StreamManager) -> Optional[ChatCompletionStreamOutput]:
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            return None

        stream_manager.delta = data['choices'][0]['delta']['content']
        stream_manager.finish_reason = data['choices'][0].get('finish_reason') or None
        stream_manager.extra['id'] = data['id']
        usage = data.get('usage')
        if usage:
            cost = self._calculate_cost(usage['total_tokens'])
            stream_manager.extra['usage'] = usage
            stream_manager.cost = cost
        return stream_manager.build_stream_output()

    def _calculate_cost(self, total_tokens: int) -> float | None:
        if self.model == 'Baichuan2-53B':
            eight_am = 8
            if 0 <= datetime.now().hour < eight_am:
                return (total_tokens * 0.01) / 1000
            return (total_tokens * 0.02) / 1000
        if self.model == 'Baichuan2-Turbo':
            return (total_tokens * 0.008) / 1000
        if self.model == 'Baichuan2-Turbo-192k':
            return (total_tokens * 0.016) / 1000
        return None
