from __future__ import annotations

import os
from typing import Any, ClassVar, Literal, Optional

from pydantic import Field, PositiveInt, field_validator
from typing_extensions import Annotated, Self, TypedDict, Unpack, override

from generate.chat_completion.http_chat import (
    HttpChatModel,
    HttpModelInitKwargs,
    HttpResponse,
    HttpxPostKwargs,
    UnexpectedResponseError,
)
from generate.chat_completion.message import (
    AssistantMessage,
    Message,
    Messages,
    MessageTypeError,
    UserMessage,
)
from generate.chat_completion.model_output import ChatCompletionModelOutput, FinishStream, Stream
from generate.parameters import ModelParameters
from generate.types import Probability, Temperature


class MinimaxMessage(TypedDict):
    sender_type: Literal['USER', 'BOT']
    text: str


class RoleMeta(TypedDict):
    user_name: str
    bot_name: str


class MinimaxChatParameters(ModelParameters):
    prompt: str = 'MM智能助理是一款由MiniMax自研的，没有调用其他产品的接口的大型语言模型。MiniMax是一家中国科技公司，一直致力于进行大模型相关的研究。'
    role_meta: RoleMeta = {'user_name': '用户', 'bot_name': 'MM智能助理'}
    beam_width: Optional[Annotated[int, Field(ge=1, le=4)]] = None
    temperature: Optional[Temperature] = None
    top_p: Optional[Probability] = None
    max_tokens: Optional[Annotated[PositiveInt, Field(serialization_alias='tokens_to_generate')]] = None
    skip_info_mask: Optional[bool] = None
    continue_last_message: Optional[bool] = None

    @field_validator('temperature', 'top_p', mode='after')
    @classmethod
    def zero_is_not_valid(cls, value: float) -> float:
        if value == 0:
            return 0.01
        return value


def convert_to_minimax_message(message: Message) -> MinimaxMessage:
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


class MinimaxChat(HttpChatModel[MinimaxChatParameters]):
    model_type: ClassVar[str] = 'minimax'
    default_api_base: ClassVar[str] = 'https://api.minimax.chat/v1/text/chatcompletion'

    def __init__(
        self,
        model: str = 'abab5.5-chat',
        group_id: str | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        system_prompt: str | None = None,
        parameters: MinimaxChatParameters | None = None,
        **kwagrs: Unpack[HttpModelInitKwargs],
    ) -> None:
        parameters = parameters or MinimaxChatParameters()
        if system_prompt is not None:
            parameters.prompt = system_prompt
        super().__init__(parameters=parameters, **kwagrs)

        self.model = model
        self.group_id = group_id or os.environ['MINIMAX_GROUP_ID']
        self.api_key = api_key or os.environ['MINIMAX_API_KEY']
        self.api_base = api_base or self.default_api_base

    @override
    def _get_request_parameters(self, messages: Messages, parameters: MinimaxChatParameters) -> HttpxPostKwargs:
        minimax_messages = [convert_to_minimax_message(message) for message in messages]
        parameters_dict = parameters.model_dump(exclude_none=True, by_alias=True)
        if 'temperature' in parameters_dict:
            parameters_dict['temperature'] = max(0.01, parameters_dict['temperature'])
        json_data = {
            'model': self.model,
            'messages': minimax_messages,
            **parameters_dict,
        }
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
        }
        return {
            'url': self.api_base,
            'json': json_data,
            'headers': headers,
            'params': {'GroupId': self.group_id},
        }

    @override
    def _parse_reponse(self, response: HttpResponse) -> ChatCompletionModelOutput:
        try:
            messages = [AssistantMessage(content=response['choices'][0]['text'])]
            return ChatCompletionModelOutput(
                chat_model_id=self.model_id,
                messages=messages,
                finish_reason=response['choices'][0]['finish_reason'],
                usage=response['usage'],
                cost=self.calculate_cost(response['usage']),
                extra={
                    'logprobes': response['choices'][0]['logprobes'],
                    'input_sensitive': False,
                    'output_sensitive': False,
                },
            )
        except (KeyError, IndexError, TypeError) as e:
            raise UnexpectedResponseError(response) from e

    @override
    def _get_stream_request_parameters(self, messages: Messages, parameters: MinimaxChatParameters) -> HttpxPostKwargs:
        http_parameters = self._get_request_parameters(messages, parameters)
        http_parameters['json']['stream'] = True
        http_parameters['json']['use_standard_sse'] = True
        return http_parameters

    @override
    def _parse_stream_response(self, response: HttpResponse) -> Stream:
        delta = response['choices'][0]['delta']
        if response['reply']:
            return FinishStream(
                delta=delta,
                finish_reason=response['choices'][0]['finish_reason'],
                usage=response['usage'],
                cost=self.calculate_cost(response['usage']),
                extra={
                    'logprobes': response['choices'][0]['logprobes'],
                    'input_sensitive': False,
                    'output_sensitive': False,
                },
            )
        return Stream(delta=delta, control='continue')

    def calculate_cost(self, usage: dict[str, int]) -> float:
        return 0.015 * (usage['total_tokens'] / 1000)

    @property
    @override
    def name(self) -> str:
        return self.model

    @classmethod
    @override
    def from_name(cls, name: str, **kwargs: Any) -> Self:
        return cls(model=name, **kwargs)
