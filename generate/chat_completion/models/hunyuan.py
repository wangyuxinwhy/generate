from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
import uuid
from typing import Any, AsyncIterator, ClassVar, Iterator, Literal, Optional

from typing_extensions import Self, TypedDict, Unpack, override

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
from generate.chat_completion.model_output import ChatCompletionOutput, ChatCompletionStreamOutput, Stream
from generate.http import (
    HttpClient,
    HttpxPostKwargs,
    ResponseValue,
    UnexpectedResponseError,
)
from generate.model import ModelParameters, ModelParametersDict
from generate.platforms.hunyuan import HunyuanSettings
from generate.types import Probability, Temperature


class HunyuanMessage(TypedDict):
    role: Literal['user', 'assistant']
    content: str


class HunyuanChatParameters(ModelParameters):
    temperature: Optional[Temperature] = None
    top_p: Optional[Probability] = None


class HunyuanChatParametersDict(ModelParametersDict, total=False):
    temperature: Optional[Temperature]
    top_p: Optional[Probability]


def _convert_message_to_hunyuan_message(message: Message) -> HunyuanMessage:
    if isinstance(message, UserMessage):
        return {'role': 'user', 'content': message.content}
    if isinstance(message, AssistantMessage):
        return {'role': 'assistant', 'content': message.content}
    raise MessageTypeError(message, (UserMessage, AssistantMessage))


def _convert_messages(messages: Messages) -> list[HunyuanMessage]:
    if isinstance(system_message := messages[0], SystemMessage):
        prepend_messages = [UserMessage(content=system_message.content), AssistantMessage(content='好的')]
        messages = prepend_messages + messages[1:]
    return [_convert_message_to_hunyuan_message(message) for message in messages]


class HunyuanChat(RemoteChatCompletionModel):
    model_type: ClassVar[str] = 'hunyuan'

    parameters: HunyuanChatParameters
    settings: HunyuanSettings

    def __init__(
        self,
        parameters: HunyuanChatParameters | None = None,
        settings: HunyuanSettings | None = None,
        http_client: HttpClient | None = None,
    ) -> None:
        parameters = parameters or HunyuanChatParameters()
        settings = settings or HunyuanSettings()  # type: ignore
        http_client = http_client or HttpClient()
        super().__init__(parameters=parameters, settings=settings, http_client=http_client)

    def _get_request_parameters(self, messages: Messages, parameters: HunyuanChatParameters) -> HttpxPostKwargs:
        hunyuan_messages = _convert_messages(messages)
        json_dict = self.generate_json_dict(hunyuan_messages, parameters)
        signature = self.generate_signature(self.generate_sign_parameters(json_dict))
        headers = {
            'Content-Type': 'application/json',
            'Authorization': signature,
        }
        return {
            'url': self.settings.completion_api,
            'headers': headers,
            'json': json_dict,
        }

    @override
    def generate(self, prompt: Prompt, **kwargs: Unpack[HunyuanChatParametersDict]) -> ChatCompletionOutput:
        messages = ensure_messages(prompt)
        parameters = self.parameters.clone_with_changes(**kwargs)
        request_parameters = self._get_request_parameters(messages, parameters)
        response = self.http_client.post(request_parameters=request_parameters)
        return self._parse_reponse(response.json())

    @override
    async def async_generate(self, prompt: Prompt, **kwargs: Unpack[HunyuanChatParametersDict]) -> ChatCompletionOutput:
        messages = ensure_messages(prompt)
        parameters = self.parameters.clone_with_changes(**kwargs)
        request_parameters = self._get_request_parameters(messages, parameters)
        response = await self.http_client.async_post(request_parameters=request_parameters)
        return self._parse_reponse(response.json())

    def _parse_reponse(self, response: ResponseValue) -> ChatCompletionOutput:
        if response.get('error'):
            raise UnexpectedResponseError(response)
        return ChatCompletionOutput(
            model_info=self.model_info,
            message=AssistantMessage(content=response['choices'][0]['messages']['content']),
            finish_reason=response['choices'][0]['finish_reason'],
            cost=self.calculate_cost(response['usage']),
            extra={'usage': response['usage']},
        )

    def _get_stream_request_parameters(self, messages: Messages, parameters: HunyuanChatParameters) -> HttpxPostKwargs:
        hunyuan_messages = _convert_messages(messages)
        json_dict = self.generate_json_dict(hunyuan_messages, parameters, stream=True)
        signature = self.generate_signature(self.generate_sign_parameters(json_dict))
        headers = {
            'Content-Type': 'application/json',
            'Authorization': signature,
        }
        return {
            'url': self.settings.completion_api,
            'headers': headers,
            'json': json_dict,
        }

    @override
    def stream_generate(
        self, prompt: Prompt, **kwargs: Unpack[HunyuanChatParametersDict]
    ) -> Iterator[ChatCompletionStreamOutput]:
        messages = ensure_messages(prompt)
        parameters = self.parameters.clone_with_changes(**kwargs)
        request_parameters = self._get_stream_request_parameters(messages, parameters)
        message = AssistantMessage(content='')
        is_start = True
        for line in self.http_client.stream_post(request_parameters=request_parameters):
            output = self._parse_stream_line(line, message, is_start)
            is_start = False
            yield output

    @override
    async def async_stream_generate(
        self, prompt: Prompt, **kwargs: Unpack[HunyuanChatParametersDict]
    ) -> AsyncIterator[ChatCompletionStreamOutput]:
        messages = ensure_messages(prompt)
        parameters = self.parameters.clone_with_changes(**kwargs)
        request_parameters = self._get_stream_request_parameters(messages, parameters)
        message = AssistantMessage(content='')
        is_start = True
        async for line in self.http_client.async_stream_post(request_parameters=request_parameters):
            output = self._parse_stream_line(line, message, is_start)
            is_start = False
            yield output

    def _parse_stream_line(self, line: str, message: AssistantMessage, is_start: bool) -> ChatCompletionStreamOutput:
        parsed_line = json.loads(line)
        message_dict = parsed_line['choices'][0]
        delta = message_dict['delta']['content']
        message.content += delta
        if message_dict['finish_reason']:
            return ChatCompletionStreamOutput(
                model_info=self.model_info,
                message=message,
                finish_reason=message_dict['finish_reason'],
                cost=self.calculate_cost(parsed_line['usage']),
                stream=Stream(delta=delta, control='finish'),
                extra={'usage': parsed_line['usage']},
            )
        return ChatCompletionStreamOutput(
            model_info=self.model_info,
            message=message,
            finish_reason=None,
            stream=Stream(delta=delta, control='start' if is_start else 'continue'),
        )

    def generate_json_dict(
        self, messages: list[HunyuanMessage], parameters: HunyuanChatParameters, stream: bool = False
    ) -> dict[str, Any]:
        timestamp = int(time.time()) + 10000
        json_dict = {
            'app_id': self.settings.app_id,
            'secret_id': self.settings.secret_id.get_secret_value(),
            'query_id': 'query_id_' + str(uuid.uuid4()),
            'messages': messages,
            'timestamp': timestamp,
            'expired': timestamp + 24 * 60 * 60,
            'stream': int(stream),
        }
        json_dict.update(parameters.custom_model_dump())
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
        sign_str = self.settings.sign_api + '?'
        for key in sort_dict:
            sign_str = sign_str + key + '=' + str(sign_parameters[key]) + '&'
        sign_str = sign_str[:-1]
        hmacstr = hmac.new(
            self.settings.secret_key.get_secret_value().encode('utf-8'), sign_str.encode('utf-8'), hashlib.sha1
        ).digest()
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
    def from_name(cls, name: str) -> Self:
        if name != 'v1':
            raise ValueError('Unknown name: {}, only support v1'.format(name))
        return cls()
