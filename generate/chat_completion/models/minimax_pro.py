from __future__ import annotations

import json
import os
from typing import Any, AsyncIterator, ClassVar, Dict, Iterator, List, Literal, Optional

from pydantic import Field, PositiveInt, field_validator, model_validator
from typing_extensions import Annotated, NotRequired, Self, TypedDict, Unpack, override

from generate.chat_completion.base import ChatCompletionModel
from generate.chat_completion.function_call import FunctionJsonSchema
from generate.chat_completion.message import (
    AssistantMessage,
    FunctionCall,
    FunctionCallMessage,
    FunctionMessage,
    Message,
    Messages,
    MessageTypeError,
    MessageValueError,
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


class BotSettingDict(TypedDict):
    bot_name: str
    content: str


class GlyphDict(TypedDict):
    type: str
    raw_glpyh: str
    json_properties: Dict[str, Any]


class ReplyConstrainsDict(TypedDict):
    sender_type: str
    sender_name: str
    glyph: NotRequired[GlyphDict]


class MinimaxFunctionCall(TypedDict):
    name: str
    arguments: str


class MinimaxProMessage(TypedDict):
    sender_type: Literal['USER', 'BOT', 'FUNCTION']
    sender_name: str
    text: str
    function_call: NotRequired[MinimaxFunctionCall]


class MinimaxProChatParameters(ModelParameters):
    reply_constraints: ReplyConstrainsDict = {'sender_type': 'BOT', 'sender_name': 'MM智能助理'}
    bot_setting: List[BotSettingDict] = [
        {
            'bot_name': 'MM智能助理',
            'content': 'MM智能助理是一款由MiniMax自研的，没有调用其他产品的接口的大型语言模型。MiniMax是一家中国科技公司，一直致力于进行大模型相关的研究。',
        }
    ]
    temperature: Optional[Temperature] = None
    top_p: Optional[Probability] = None
    max_tokens: Annotated[Optional[PositiveInt], Field(serialization_alias='tokens_to_generate')] = None
    mask_sensitive_info: Optional[bool] = None
    sample_messages: Optional[List[MinimaxProMessage]] = None
    functions: Optional[List[FunctionJsonSchema]] = None
    plugins: Optional[List[str]] = None

    @model_validator(mode='after')
    def check_bot_name(self) -> Self:
        names: set[str] = {bot_setting['bot_name'] for bot_setting in self.bot_setting}
        if (sender_name := self.reply_constraints['sender_name']) not in names:
            raise ValueError(f'reply_constraints sender_name {sender_name} must be in bot_setting names: {names}')
        return self

    @field_validator('temperature', 'top_p', mode='after')
    @classmethod
    def zero_is_not_valid(cls, value: float) -> float:
        if value == 0:
            return 0.01
        return value

    @property
    def bot_name(self) -> str | None:
        if len(self.bot_setting) == 1:
            return self.bot_setting[0]['bot_name']
        return None

    def set_system_prompt(self, system_prompt: str) -> None:
        if len(self.bot_setting) == 1:
            self.bot_setting[0]['content'] = system_prompt
        else:
            raise ValueError('set system_prompt is not supported when bot_setting has more than one bot')

    def set_bot_name(self, bot_name: str) -> None:
        if len(self.bot_setting) == 1:
            self.bot_setting[0]['bot_name'] = bot_name
            self.reply_constraints['sender_name'] = bot_name
        else:
            raise ValueError('set bot_name is not supported when bot_setting has more than one bot')


def convert_to_minimax_pro_message(
    message: Message, default_bot_name: str | None = None, default_user_name: str = '用户'
) -> MinimaxProMessage:
    if isinstance(message, UserMessage):
        sender_name = message.name or default_user_name
        return {'sender_type': 'USER', 'sender_name': sender_name, 'text': message.content}

    if isinstance(message, AssistantMessage):
        sender_name = message.name or default_bot_name
        if sender_name is None:
            raise MessageValueError(message, 'bot name is required')
        return {
            'sender_type': 'BOT',
            'sender_name': sender_name,
            'text': message.content,
        }

    if isinstance(message, FunctionCallMessage):
        sender_name = message.name or default_bot_name
        if sender_name is None:
            raise MessageValueError(message, 'bot name is required')
        return {
            'sender_type': 'BOT',
            'sender_name': sender_name,
            'text': '',
            'function_call': {
                'name': message.content.name,
                'arguments': message.content.arguments,
            },
        }

    if isinstance(message, FunctionMessage):
        if message.name is None:
            raise MessageValueError(message, 'function name is required')
        return {
            'sender_type': 'FUNCTION',
            'sender_name': message.name,
            'text': message.content,
        }

    raise MessageTypeError(message, allowed_message_type=(UserMessage, AssistantMessage, FunctionMessage, FunctionCallMessage))


class MinimaxProChat(ChatCompletionModel[MinimaxProChatParameters], HttpMixin):
    model_type: ClassVar[str] = 'minimax_pro'
    default_api_base: ClassVar[str] = 'https://api.minimax.chat/v1/text/chatcompletion_pro'

    def __init__(
        self,
        model: str = 'abab5.5-chat',
        group_id: str | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        bot_name: str | None = None,
        system_prompt: str | None = None,
        default_user_name: str = '用户',
        parameters: MinimaxProChatParameters | None = None,
        **kwargs: Unpack[HttpClientInitKwargs],
    ) -> None:
        parameters = parameters or MinimaxProChatParameters()
        self.default_user_name = default_user_name
        if system_prompt is not None:
            parameters.set_system_prompt(system_prompt)
        if bot_name is not None:
            parameters.set_bot_name(bot_name)
        super().__init__(parameters=parameters)

        self.model = model
        self.group_id = group_id or os.environ['MINIMAX_GROUP_ID']
        self.api_key = api_key or os.environ['MINIMAX_API_KEY']
        self.api_base = api_base or self.default_api_base
        self.http_client = HttpClient(**kwargs)
        self.http_stream_client = HttpStreamClient(**kwargs)

    def _get_request_parameters(self, messages: Messages, parameters: MinimaxProChatParameters) -> HttpxPostKwargs:
        minimax_pro_messages = [
            convert_to_minimax_pro_message(
                message, default_bot_name=parameters.bot_name, default_user_name=self.default_user_name
            )
            for message in messages
        ]
        parameters_dict = parameters.model_dump(exclude_none=True, by_alias=True)
        json_data = {'model': self.model, 'messages': minimax_pro_messages, **parameters_dict}
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

    def _completion(self, messages: Messages, parameters: MinimaxProChatParameters) -> ChatCompletionOutput:
        request_parameters = self._get_request_parameters(messages, parameters)
        response = self.http_client.post(request_parameters=request_parameters)
        return self._parse_reponse(response.json())

    async def _async_completion(self, messages: Messages, parameters: MinimaxProChatParameters) -> ChatCompletionOutput:
        request_parameters = self._get_request_parameters(messages, parameters)
        response = await self.http_client.async_post(request_parameters=request_parameters)
        return self._parse_reponse(response.json())

    def _parse_reponse(self, response: dict[str, Any]) -> ChatCompletionOutput:
        try:
            messages = [self._convert_to_message(i) for i in response['choices'][0]['messages']]
            finish_reason = response['choices'][0]['finish_reason']
            num_web_search = sum([i for i in response['choices'][0]['messages'] if i['sender_name'] == 'plugin_web_search'])

            return ChatCompletionOutput(
                model_info=self.model_info,
                messages=messages,
                finish_reason=finish_reason,
                cost=self.calculate_cost(response['usage'], num_web_search),
                extra={
                    'input_sensitive': response['input_sensitive'],
                    'output_sensitive': response['output_sensitive'],
                    'usage': response['usage'],
                },
            )
        except (KeyError, IndexError, TypeError) as e:
            raise UnexpectedResponseError(response) from e

    def _get_stream_request_parameters(self, messages: Messages, parameters: MinimaxProChatParameters) -> HttpxPostKwargs:
        http_parameters = self._get_request_parameters(messages, parameters)
        http_parameters['json']['stream'] = True
        return http_parameters

    def _stream_completion(
        self, messages: Messages, parameters: MinimaxProChatParameters
    ) -> Iterator[ChatCompletionStreamOutput]:
        request_parameters = self._get_stream_request_parameters(messages, parameters)
        yield ChatCompletionStreamOutput(
            model_info=self.model_info,
            stream=Stream(delta='', control='start'),
        )
        for line in self.http_stream_client.post(request_parameters=request_parameters):
            output = self._parse_stream_line(line)
            yield output
            if output.is_finish:
                break

    async def _async_stream_completion(
        self, messages: Messages, parameters: MinimaxProChatParameters
    ) -> AsyncIterator[ChatCompletionStreamOutput]:
        request_parameters = self._get_stream_request_parameters(messages, parameters)
        yield ChatCompletionStreamOutput(
            model_info=self.model_info,
            stream=Stream(delta='', control='start'),
        )
        async for line in self.http_stream_client.async_post(request_parameters=request_parameters):
            output = self._parse_stream_line(line)
            yield output
            if output.is_finish:
                break

    def _parse_stream_line(self, line: str) -> ChatCompletionStreamOutput:
        parsed_line = json.loads(line)
        delta = parsed_line['choices'][0]['messages'][0]['text']
        if parsed_line['reply']:
            return ChatCompletionStreamOutput(
                model_info=self.model_info,
                messages=[AssistantMessage(content=parsed_line['reply'])],
                finish_reason=parsed_line['choices'][0]['finish_reason'],
                cost=self.calculate_cost(parsed_line['usage']),
                extra={
                    'input_sensitive': parsed_line['input_sensitive'],
                    'output_sensitive': parsed_line['output_sensitive'],
                    'usage': parsed_line['usage'],
                },
                stream=Stream(delta=delta, control='finish'),
            )
        return ChatCompletionStreamOutput(
            model_info=self.model_info,
            stream=Stream(delta=delta, control='continue'),
        )

    @staticmethod
    def _convert_to_message(message: MinimaxProMessage) -> Message:
        if 'function_call' in message:
            return FunctionCallMessage(
                name=message['sender_name'],
                content=FunctionCall(name=message['function_call']['name'], arguments=message['function_call']['arguments']),
            )
        if message['sender_type'] == 'USER':
            return UserMessage(
                name=message['sender_name'],
                content=message['text'],
            )
        if message['sender_type'] == 'BOT':
            return AssistantMessage(
                name=message['sender_name'],
                content=message['text'],
            )
        if message['sender_type'] == 'FUNCTION':
            return FunctionMessage(
                name=message['sender_name'],
                content=message['text'],
            )
        raise ValueError(f'unknown sender_type: {message["sender_type"]}')

    def calculate_cost(self, usage: dict[str, int], num_web_search: int = 0) -> float:
        return 0.015 * (usage['total_tokens'] / 1000) + (0.03 * num_web_search)

    @property
    @override
    def name(self) -> str:
        return self.model

    @classmethod
    @override
    def from_name(cls, name: str, **kwargs: Any) -> Self:
        return cls(model=name, **kwargs)
