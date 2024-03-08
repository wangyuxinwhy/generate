from __future__ import annotations

import json
from typing import Any, AsyncIterator, ClassVar, Iterator, List, Literal, Optional

from pydantic import Field, PositiveInt
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
from generate.platforms.minimax import MinimaxSettings
from generate.types import Probability, Temperature


class MinimaxMessage(TypedDict):
    sender_type: Literal['USER', 'BOT']
    text: str


class RoleMeta(TypedDict):
    user_name: str
    bot_name: str


DEFAULT_MINIMAX_SYSTEM_PROMPT = 'MM智能助理是一款由MiniMax自研的，没有调用其他产品的接口的大型语言模型。MiniMax是一家中国科技公司，一直致力于进行大模型相关的研究。'


class MinimaxLegacyChatParameters(ModelParameters):
    system_prompt: str = Field(default=DEFAULT_MINIMAX_SYSTEM_PROMPT, serialization_alias='prompt')
    role_meta: RoleMeta = {'user_name': '用户', 'bot_name': 'MM智能助理'}
    beam_width: Optional[Annotated[int, Field(ge=1, le=4)]] = None
    temperature: Optional[Temperature] = None
    top_p: Optional[Probability] = None
    max_tokens: Optional[Annotated[PositiveInt, Field(serialization_alias='tokens_to_generate')]] = None
    skip_info_mask: Optional[bool] = None
    continue_last_message: Optional[bool] = None

    def custom_model_dump(self) -> dict[str, Any]:
        output = super().custom_model_dump()
        if 'temperature' in output:
            output['temperature'] = max(0.01, output['temperature'])
        if 'top_p' in output:
            output['top_p'] = max(0.01, output['top_p'])
        return output


class MinimaxLegacyChatParametersDict(RemoteModelParametersDict, total=False):
    system_prompt: str
    role_meta: RoleMeta
    beam_width: Optional[int]
    temperature: Optional[Temperature]
    top_p: Optional[Probability]
    max_tokens: Optional[int]
    skip_info_mask: Optional[bool]
    continue_last_message: Optional[bool]


def _convert_message_to_minimax_message(message: Message) -> MinimaxMessage:
    if isinstance(message, UserMessage):
        return {
            'sender_type': 'USER',
            'text': message.content,
        }
    if isinstance(message, AssistantMessage):
        return {
            'sender_type': 'BOT',
            'text': message.content,
        }
    raise MessageTypeError(message, (UserMessage, AssistantMessage))


def _convert_messages(messages: Messages) -> list[MinimaxMessage]:
    if isinstance(system_message := messages[0], SystemMessage):
        prepend_messages = [UserMessage(content=system_message.content), AssistantMessage(content='好的')]
        messages = prepend_messages + messages[1:]
    return [_convert_message_to_minimax_message(message) for message in messages]


class MinimaxLegacyChat(RemoteChatCompletionModel):
    model_type: ClassVar[str] = 'minimax_legacy'
    available_models: ClassVar[List[str]] = ['abab5.5-chat', 'abab5.5s-chat']

    parameters: MinimaxLegacyChatParameters
    settings: MinimaxSettings

    def __init__(
        self,
        model: str = 'abab5.5-chat',
        settings: MinimaxSettings | None = None,
        parameters: MinimaxLegacyChatParameters | None = None,
        http_client: HttpClient | None = None,
    ) -> None:
        parameters = parameters or MinimaxLegacyChatParameters()
        settings = settings or MinimaxSettings()  # type: ignore
        http_client = http_client or HttpClient()
        if not settings.group_id:
            raise ValueError(
                'group_id is required for MinimaxLegacyChat, you can set it in settings or environment variable MINIMAX_GROUP_ID'
            )
        super().__init__(model=model, parameters=parameters, settings=settings, http_client=http_client)

    @override
    def generate(self, prompt: Prompt, **kwargs: Unpack[MinimaxLegacyChatParametersDict]) -> ChatCompletionOutput:
        return super().generate(prompt, **kwargs)

    @override
    async def async_generate(self, prompt: Prompt, **kwargs: Unpack[MinimaxLegacyChatParametersDict]) -> ChatCompletionOutput:
        return await super().async_generate(prompt, **kwargs)

    @override
    def stream_generate(
        self, prompt: Prompt, **kwargs: Unpack[MinimaxLegacyChatParametersDict]
    ) -> Iterator[ChatCompletionStreamOutput]:
        yield from super().stream_generate(prompt, **kwargs)

    @override
    async def async_stream_generate(
        self, prompt: Prompt, **kwargs: Unpack[MinimaxLegacyChatParametersDict]
    ) -> AsyncIterator[ChatCompletionStreamOutput]:
        async for output in super().async_stream_generate(prompt, **kwargs):
            yield output

    @override
    def _get_request_parameters(
        self, prompt: Prompt, stream: bool = False, **kwargs: Unpack[MinimaxLegacyChatParametersDict]
    ) -> HttpxPostKwargs:
        messages = ensure_messages(prompt)
        parameters = self.parameters.clone_with_changes(**kwargs)
        minimax_messages = _convert_messages(messages)
        parameters_dict = parameters.custom_model_dump()
        json_data = {
            'model': self.model,
            'messages': minimax_messages,
            **parameters_dict,
        }
        if stream:
            json_data['stream'] = True
            json_data['use_standard_sse'] = True

        headers = {
            'Authorization': f'Bearer {self.settings.api_key.get_secret_value()}',
            'Content-Type': 'application/json',
        }
        return {
            'url': self.settings.api_base + '/text/chatcompletion',
            'json': json_data,
            'headers': headers,
            'params': {'GroupId': self.settings.group_id},
        }

    @override
    def _process_reponse(self, response: ResponseValue) -> ChatCompletionOutput:
        try:
            return ChatCompletionOutput(
                model_info=self.model_info,
                message=AssistantMessage(content=response['choices'][0]['text']),
                finish_reason=response['choices'][0]['finish_reason'],
                cost=self._calculate_cost(response['usage']),
                extra={
                    'logprobes': response['choices'][0]['logprobes'],
                    'input_sensitive': False,
                    'output_sensitive': False,
                    'usage': response['usage'],
                },
            )
        except (KeyError, IndexError, TypeError) as e:
            raise UnexpectedResponseError(response) from e

    @override
    def _process_stream_line(self, line: str, stream_manager: StreamManager) -> ChatCompletionStreamOutput | None:
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            return None
        stream_manager.delta = data['choices'][0]['delta']

        if data['reply']:
            stream_manager.finish_reason = data['choices'][0]['finish_reason']
            extra = {
                'logprobes': data['choices'][0]['logprobes'],
                'input_sensitive': False,
                'output_sensitive': False,
                'usage': data['usage'],
            }
            stream_manager.extra.update(extra)
            stream_manager.cost = self._calculate_cost(data['usage'])
        return stream_manager.build_stream_output()

    def _calculate_cost(self, usage: dict[str, int]) -> float:
        return 0.015 * (usage['total_tokens'] / 1000)
