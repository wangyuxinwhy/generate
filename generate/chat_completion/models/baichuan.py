from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime
from typing import Any, ClassVar, Literal, Optional, TypedDict

from pydantic import Field
from typing_extensions import Annotated, Self, Unpack, override

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


class BaichuanMessage(TypedDict):
    role: Literal['user', 'assistant']
    content: str


class BaichuanChatParameters(ModelParameters):
    temperature: Optional[Temperature] = None
    top_k: Optional[Annotated[int, Field(ge=0)]] = None
    top_p: Optional[Probability] = None
    with_search_enhance: Optional[bool] = None


def convert_to_baichuan_message(message: Message) -> BaichuanMessage:
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


class BaichuanChat(HttpChatModel[BaichuanChatParameters]):
    model_type: ClassVar[str] = 'baichuan'
    parse_stream_strategy: ClassVar[str] = 'basic'
    default_api_base: ClassVar[str] = 'https://api.baichuan-ai.com/v1/chat'
    default_stream_api_base: ClassVar[str] = 'https://api.baichuan-ai.com/v1/stream/chat'

    def __init__(
        self,
        model: str = 'Baichuan2-53B',
        api_key: str | None = None,
        secret_key: str | None = None,
        api_base: str | None = None,
        stream_api_base: str | None = None,
        parameters: BaichuanChatParameters | None = None,
        **kwargs: Unpack[HttpModelInitKwargs],
    ) -> None:
        parameters = parameters or BaichuanChatParameters()
        super().__init__(parameters=parameters, **kwargs)
        self.model = model
        self.api_key = api_key or os.environ['BAICHUAN_API_KEY']
        self.secret_key = secret_key or os.environ['BAICHUAN_SECRET_KEY']
        self.api_base = api_base or self.default_api_base
        self.api_base.rstrip('/')
        self.stream_api_base = stream_api_base or self.default_stream_api_base
        self.stream_api_base.rstrip('/')

    @override
    def _get_request_parameters(self, messages: Messages, parameters: BaichuanChatParameters) -> HttpxPostKwargs:
        baichuan_messages: list[BaichuanMessage] = [convert_to_baichuan_message(message) for message in messages]
        data = {
            'model': self.model,
            'messages': baichuan_messages,
        }
        parameters_dict = parameters.model_dump(exclude_none=True)
        if parameters_dict:
            data['parameters'] = parameters_dict
        time_stamp = int(time.time())
        signature = self.calculate_md5(self.secret_key + json.dumps(data) + str(time_stamp))

        headers = {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer ' + self.api_key,
            'X-BC-Timestamp': str(time_stamp),
            'X-BC-Signature': signature,
            'X-BC-Sign-Algo': 'MD5',
            'X-BC-Request-Id': 'your requestId',
        }
        return {
            'url': self.api_base,
            'headers': headers,
            'json': data,
        }

    @override
    def _get_stream_request_parameters(self, messages: Messages, parameters: BaichuanChatParameters) -> HttpxPostKwargs:
        http_parameters = self._get_request_parameters(messages, parameters)
        http_parameters['url'] = self.stream_api_base
        return http_parameters

    @override
    def _parse_stream_response(self, response: HttpResponse) -> Stream:
        message = response['data']['messages'][-1]

        if message['finish_reason']:
            return FinishStream(
                delta=message['content'],
                control='finish',
                finish_reason=message['finish_reason'],
                usage=response['usage'],
                cost=self.calculate_cost(response['usage']),
            )

        return Stream(delta=message['content'], control='continue')

    @staticmethod
    def calculate_md5(input_string: str) -> str:
        md5 = hashlib.md5()
        md5.update(input_string.encode('utf-8'))
        return md5.hexdigest()

    @override
    def _parse_reponse(self, response: HttpResponse) -> ChatCompletionModelOutput:
        try:
            text = response['data']['messages'][-1]['content']
            finish_reason = response['data']['messages'][-1]['finish_reason']
            return ChatCompletionModelOutput(
                chat_model_id=self.model_id,
                messages=[AssistantMessage(content=text)],
                finish_reason=finish_reason,
                usage=response['usage'],
                cost=self.calculate_cost(response['usage']),
            )
        except (KeyError, IndexError) as e:
            raise UnexpectedResponseError(response) from e

    def calculate_cost(self, usage: dict[str, int]) -> float | None:
        if self.name == 'Baichuan2-53B':
            eight_am = 8
            if 0 <= datetime.now().hour < eight_am:
                return (usage['total_tokens'] * 0.01) / 1000
            return (usage['total_tokens'] * 0.02) / 1000
        return None

    @property
    def name(self) -> str:
        return self.model

    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> Self:
        return cls(model=name, **kwargs)
