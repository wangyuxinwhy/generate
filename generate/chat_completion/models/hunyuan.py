from __future__ import annotations

import base64
import hashlib
import hmac
import os
import time
import uuid
from typing import Any, ClassVar, Literal, Optional

from typing_extensions import Self, TypedDict, Unpack, override

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


class HunyuanMessage(TypedDict):
    role: Literal['user', 'assistant']
    content: str


class HunyuanChatParameters(ModelParameters):
    temperature: Optional[Temperature] = None
    top_p: Optional[Probability] = None


def convert_to_hunyuan_message(message: Message) -> HunyuanMessage:
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


class HunyuanChat(HttpChatModel[HunyuanChatParameters]):
    model_type: ClassVar[str] = 'hunyuan'
    default_api: ClassVar[str] = 'https://hunyuan.cloud.tencent.com/hyllm/v1/chat/completions'
    default_sign_api: ClassVar[str] = 'hunyuan.cloud.tencent.com/hyllm/v1/chat/completions'

    def __init__(
        self,
        app_id: int | None = None,
        secret_id: str | None = None,
        secret_key: str | None = None,
        api: str | None = None,
        sign_api: str | None = None,
        parameters: HunyuanChatParameters | None = None,
        **kwargs: Unpack[HttpModelInitKwargs],
    ) -> None:
        parameters = parameters or HunyuanChatParameters()
        super().__init__(parameters=parameters, **kwargs)
        self.app_id = app_id or int(os.environ['HUNYUAN_APP_ID'])
        self.secret_id = secret_id or os.environ['HUNYUAN_SECRET_ID']
        self.secret_key = secret_key or os.environ['HUNYUAN_SECRET_KEY']
        self.api = api or self.default_api
        self.sign_api = sign_api or self.default_sign_api

    @override
    def _get_request_parameters(self, messages: Messages, parameters: HunyuanChatParameters) -> HttpxPostKwargs:
        hunyuan_messages = [convert_to_hunyuan_message(message) for message in messages]
        json_dict = self.generate_json_dict(hunyuan_messages, parameters)
        signature = self.generate_signature(self.generate_sign_parameters(json_dict))
        headers = {
            'Content-Type': 'application/json',
            'Authorization': signature,
        }
        return {
            'url': self.api,
            'headers': headers,
            'json': json_dict,
        }

    @override
    def _get_stream_request_parameters(self, messages: Messages, parameters: HunyuanChatParameters) -> HttpxPostKwargs:
        hunyuan_messages = [convert_to_hunyuan_message(message) for message in messages]
        json_dict = self.generate_json_dict(hunyuan_messages, parameters, stream=True)
        signature = self.generate_signature(self.generate_sign_parameters(json_dict))
        headers = {
            'Content-Type': 'application/json',
            'Authorization': signature,
        }
        return {
            'url': self.api,
            'headers': headers,
            'json': json_dict,
        }

    @override
    def _parse_stream_response(self, response: HttpResponse) -> Stream:
        message = response['choices'][0]
        if message['finish_reason']:
            return FinishStream(
                delta=message['delta']['content'],
                control='finish',
                usage=response['usage'],
                finish_reason=message['finish_reason'],
                cost=self.calculate_cost(response['usage']),
            )
        return Stream(delta=message['delta']['content'], control='continue')

    @override
    def _parse_reponse(self, response: HttpResponse) -> ChatCompletionModelOutput:
        if response.get('error'):
            raise UnexpectedResponseError(response)
        messages = [AssistantMessage(content=response['choices'][0]['messages']['content'])]
        return ChatCompletionModelOutput(
            chat_model_id=self.model_id,
            messages=messages,
            finish_reason=response['choices'][0]['finish_reason'],
            usage=response['usage'],
            cost=self.calculate_cost(response['usage']),
        )

    def generate_json_dict(
        self, messages: list[HunyuanMessage], parameters: HunyuanChatParameters, stream: bool = False
    ) -> dict[str, Any]:
        timestamp = int(time.time()) + 10000
        json_dict: dict[str, Any] = {
            'app_id': self.app_id,
            'secret_id': self.secret_id,
            'query_id': 'query_id_' + str(uuid.uuid4()),
            'messages': messages,
            'timestamp': timestamp,
            'expired': timestamp + 24 * 60 * 60,
            'stream': int(stream),
        }
        json_dict.update(parameters.model_dump(exclude_none=True))
        return json_dict

    @staticmethod
    def generate_sign_parameters(json_dict: dict[str, Any]) -> dict[str, Any]:
        params = {
            'app_id': json_dict['app_id'],
            'secret_id': json_dict['secret_id'],
            'query_id': json_dict['query_id'],
            'stream': json_dict['stream'],
        }
        if 'temperature' in json_dict:
            params['temperature'] = f'{json_dict["temperature"]:g}'
        if 'top_p' in json_dict:
            params['top_p'] = f'{json_dict["top_p"]:g}'
        message_str = ','.join(
            ['{{"role":"{}","content":"{}"}}'.format(message['role'], message['content']) for message in json_dict['messages']]
        )
        message_str = '[{}]'.format(message_str)
        params['messages'] = message_str
        params['timestamp'] = str(json_dict['timestamp'])
        params['expired'] = str(json_dict['expired'])
        return params

    def generate_signature(self, sign_parameters: dict[str, Any]) -> str:
        sort_dict = sorted(sign_parameters.keys())
        sign_str = self.default_sign_api + '?'
        for key in sort_dict:
            sign_str = sign_str + key + '=' + str(sign_parameters[key]) + '&'
        sign_str = sign_str[:-1]
        hmacstr = hmac.new(self.secret_key.encode('utf-8'), sign_str.encode('utf-8'), hashlib.sha1).digest()
        signature = base64.b64encode(hmacstr)
        return signature.decode('utf-8')

    def calculate_cost(self, usage: dict[str, Any]) -> float:
        return (usage['total_tokens'] / 1000) * 0.1

    @property
    @override
    def name(self) -> str:
        return 'v1'

    @classmethod
    @override
    def from_name(cls, name: str, **kwargs: Any) -> Self:
        if name != 'v1':
            raise ValueError('Unknown name: {}, only support v1'.format(name))
        return cls(**kwargs)
