from __future__ import annotations

import json
import os
from typing import Any, AsyncIterator, ClassVar, Iterator, Literal, Optional

from pydantic import Field, PositiveInt, field_validator
from typing_extensions import Annotated, Self, TypedDict, Unpack, override

from generate.chat_completion import ChatCompletionModel
from generate.chat_completion.message import (
    AssistantMessage,
    Message,
    Messages,
    MessageTypeError,
    UserMessage,
)
from generate.chat_completion.model_output import ChatCompletionOutput, ChatCompletionStreamOutput, Stream
from generate.http import (
    HttpClient,
    HttpClientInitKwargs,
    HttpMixin,
    HttpStreamClient,
    HttpxPostKwargs,
    UnexpectedResponseError,
)
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


class MinimaxChat(ChatCompletionModel[MinimaxChatParameters], HttpMixin):
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
        **kwagrs: Unpack[HttpClientInitKwargs],
    ) -> None:
        parameters = parameters or MinimaxChatParameters()
        if system_prompt is not None:
            parameters.prompt = system_prompt
        super().__init__(parameters=parameters)

        self.model = model
        self.group_id = group_id or os.environ['MINIMAX_GROUP_ID']
        self.api_key = api_key or os.environ['MINIMAX_API_KEY']
        self.api_base = api_base or self.default_api_base
        self.http_client = HttpClient(**kwagrs)
        self.http_stream_client = HttpStreamClient(**kwagrs)

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

    def _completion(self, messages: Messages, parameters: MinimaxChatParameters) -> ChatCompletionOutput:
        request_parameters = self._get_request_parameters(messages, parameters)
        response = self.http_client.post(request_parameters=request_parameters)
        return self._parse_reponse(response.json())

    async def _async_completion(self, messages: Messages, parameters: MinimaxChatParameters) -> ChatCompletionOutput:
        request_parameters = self._get_request_parameters(messages, parameters)
        response = await self.http_client.async_post(request_parameters=request_parameters)
        return self._parse_reponse(response.json())

    def _parse_reponse(self, response: dict[str, Any]) -> ChatCompletionOutput:
        try:
            messages = [AssistantMessage(content=response['choices'][0]['text'])]
            return ChatCompletionOutput(
                model_info=self.model_info,
                messages=messages,
                finish_reason=response['choices'][0]['finish_reason'],
                cost=self.calculate_cost(response['usage']),
                extra={
                    'logprobes': response['choices'][0]['logprobes'],
                    'input_sensitive': False,
                    'output_sensitive': False,
                    'usage': response['usage'],
                },
            )
        except (KeyError, IndexError, TypeError) as e:
            raise UnexpectedResponseError(response) from e

    def _get_stream_request_parameters(self, messages: Messages, parameters: MinimaxChatParameters) -> HttpxPostKwargs:
        http_parameters = self._get_request_parameters(messages, parameters)
        http_parameters['json']['stream'] = True
        http_parameters['json']['use_standard_sse'] = True
        return http_parameters

    def _stream_completion(self, messages: Messages, parameters: MinimaxChatParameters) -> Iterator[ChatCompletionStreamOutput]:
        request_parameters = self._get_stream_request_parameters(messages, parameters)
        yield ChatCompletionStreamOutput(
            model_info=self.model_info,
            stream=Stream(delta='', control='start'),
        )
        reply = ''
        for line in self.http_stream_client.post(request_parameters=request_parameters):
            output = self._parse_stream_line(line)
            reply += output.stream.delta
            if output.is_finish:
                output.messages = [AssistantMessage(content=reply)]
            yield output
            if output.is_finish:
                break

    async def _async_stream_completion(
        self, messages: Messages, parameters: MinimaxChatParameters
    ) -> AsyncIterator[ChatCompletionStreamOutput]:
        request_parameters = self._get_stream_request_parameters(messages, parameters)
        yield ChatCompletionStreamOutput(
            model_info=self.model_info,
            stream=Stream(delta='', control='start'),
        )
        reply = ''
        async for line in self.http_stream_client.async_post(request_parameters=request_parameters):
            output = self._parse_stream_line(line)
            reply += output.stream.delta
            if output.is_finish:
                output.messages = [AssistantMessage(content=reply)]
            yield output
            if output.is_finish:
                break

    def _parse_stream_line(self, line: str) -> ChatCompletionStreamOutput:
        parsed_line = json.loads(line)
        delta = parsed_line['choices'][0]['delta']
        if parsed_line['reply']:
            return ChatCompletionStreamOutput(
                model_info=self.model_info,
                finish_reason=parsed_line['choices'][0]['finish_reason'],
                cost=self.calculate_cost(parsed_line['usage']),
                extra={
                    'logprobes': parsed_line['choices'][0]['logprobes'],
                    'input_sensitive': False,
                    'output_sensitive': False,
                    'usage': parsed_line['usage'],
                },
                stream=Stream(delta=delta, control='finish'),
            )
        return ChatCompletionStreamOutput(
            model_info=self.model_info,
            stream=Stream(delta=delta, control='continue'),
        )

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
