from __future__ import annotations

import hashlib
import json
import time
import uuid
from datetime import datetime
from typing import AsyncIterator, ClassVar, Iterator, Literal, Optional

from pydantic import Field
from typing_extensions import Annotated, Self, TypedDict, Unpack, override

from generate.chat_completion.base import ChatCompletionModel
from generate.chat_completion.message import (
    AssistantMessage,
    Message,
    Messages,
    MessageTypeError,
    Prompt,
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


class BaichuanChatParametersDict(ModelParametersDict, total=False):
    temperature: Optional[Temperature]
    top_k: Optional[int]
    top_p: Optional[Probability]
    search: Optional[bool]


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


class BaichuanChat(ChatCompletionModel):
    model_type: ClassVar[str] = 'baichuan'

    def __init__(
        self,
        model: str = 'Baichuan2-53B',
        parameters: BaichuanChatParameters | None = None,
        settings: BaichuanSettings | None = None,
        http_client: HttpClient | None = None,
    ) -> None:
        self.model = model
        self.parameters = parameters or BaichuanChatParameters()
        self.settings = settings or BaichuanSettings()  # type: ignore
        self.http_client = http_client or HttpClient()
        self.http_client.stream_strategy = 'basic'

    def _get_request_parameters(self, messages: Messages, parameters: BaichuanChatParameters) -> HttpxPostKwargs:
        baichuan_messages: list[BaichuanMessage] = [convert_to_baichuan_message(message) for message in messages]
        data = {
            'model': self.model,
            'messages': baichuan_messages,
        }
        parameters_dict = parameters.custom_model_dump()
        if parameters_dict:
            data['parameters'] = parameters_dict
        time_stamp = int(time.time())
        signature = self.calculate_md5(self.settings.secret_key.get_secret_value() + json.dumps(data) + str(time_stamp))

        headers = {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer ' + self.settings.api_key.get_secret_value(),
            'X-BC-Timestamp': str(time_stamp),
            'X-BC-Signature': signature,
            'X-BC-Sign-Algo': 'MD5',
            'X-BC-Request-Id': str(uuid.uuid4()),
        }
        return {
            'url': self.settings.api_base,
            'headers': headers,
            'json': data,
        }

    @override
    def generate(self, prompt: Prompt, **kwargs: Unpack[BaichuanChatParametersDict]) -> ChatCompletionOutput[AssistantMessage]:
        messages = ensure_messages(prompt)
        parameters = self.parameters.update_with_validate(**kwargs)
        request_parameters = self._get_request_parameters(messages, parameters)
        response = self.http_client.post(request_parameters=request_parameters)
        return self._parse_reponse(response.json())

    @override
    async def async_generate(
        self, prompt: Prompt, **kwargs: Unpack[BaichuanChatParametersDict]
    ) -> ChatCompletionOutput[AssistantMessage]:
        messages = ensure_messages(prompt)
        parameters = self.parameters.update_with_validate(**kwargs)
        request_parameters = self._get_request_parameters(messages, parameters)
        response = await self.http_client.async_post(request_parameters=request_parameters)
        return self._parse_reponse(response.json())

    def _parse_reponse(self, response: ResponseValue) -> ChatCompletionOutput[AssistantMessage]:
        try:
            text = response['data']['messages'][-1]['content']
            finish_reason = response['data']['messages'][-1]['finish_reason']
            if not finish_reason:
                finish_reason = None
            if usage := response.get('usage'):
                cost = self.calculate_cost(usage)
                extra = {'usage': usage}
            else:
                cost = None
                extra = {}
            return ChatCompletionOutput(
                model_info=self.model_info,
                messages=[AssistantMessage(content=text)],
                finish_reason=finish_reason,
                cost=cost,
                extra=extra,
            )
        except (KeyError, IndexError) as e:
            raise UnexpectedResponseError(response) from e

    def _get_stream_request_parameters(self, messages: Messages, parameters: BaichuanChatParameters) -> HttpxPostKwargs:
        http_parameters = self._get_request_parameters(messages, parameters)
        http_parameters['url'] = self.settings.stream_api_base
        return http_parameters

    @override
    def stream_generate(
        self, prompt: Prompt, **kwargs: Unpack[BaichuanChatParametersDict]
    ) -> Iterator[ChatCompletionStreamOutput[AssistantMessage]]:
        messages = ensure_messages(prompt)
        parameters = self.parameters.update_with_validate(**kwargs)
        request_parameters = self._get_stream_request_parameters(messages, parameters)
        message = AssistantMessage(content='')
        is_start = True
        for line in self.http_client.stream_post(request_parameters=request_parameters):
            output = self._parse_stream_line(line, message, is_start)
            is_start = False
            yield output

    @override
    async def async_stream_generate(
        self, prompt: Prompt, **kwargs: Unpack[BaichuanChatParametersDict]
    ) -> AsyncIterator[ChatCompletionStreamOutput[AssistantMessage]]:
        messages = ensure_messages(prompt)
        parameters = self.parameters.update_with_validate(**kwargs)
        request_parameters = self._get_stream_request_parameters(messages, parameters)
        message = AssistantMessage(content='')
        is_start = True
        async for line in self.http_client.async_stream_post(request_parameters=request_parameters):
            output = self._parse_stream_line(line, message, is_start)
            is_start = False
            yield output

    def _parse_stream_line(
        self, line: str, message: AssistantMessage, is_start: bool
    ) -> ChatCompletionStreamOutput[AssistantMessage]:
        output = self._parse_reponse(json.loads(line))
        output_message = output.message
        if is_start:
            stream = Stream(delta=output_message.content, control='start')
        else:
            stream = Stream(delta=output_message.content, control='finish' if output.is_finish else 'continue')
        message.content += output_message.content
        output.messages = [message]
        return ChatCompletionStreamOutput[AssistantMessage](
            model_info=output.model_info,
            messages=output.messages,
            finish_reason=output.finish_reason,
            cost=output.cost,
            extra=output.extra,
            stream=stream,
        )

    @staticmethod
    def calculate_md5(input_string: str) -> str:
        md5 = hashlib.md5()
        md5.update(input_string.encode('utf-8'))
        return md5.hexdigest()

    def calculate_cost(self, usage: dict[str, int]) -> float | None:
        if self.name == 'Baichuan2-53B':
            eight_am = 8
            if 0 <= datetime.now().hour < eight_am:
                return (usage['total_tokens'] * 0.01) / 1000
            return (usage['total_tokens'] * 0.02) / 1000
        return None

    @property
    @override
    def name(self) -> str:
        return self.model

    @classmethod
    @override
    def from_name(cls, name: str) -> Self:
        return cls(model=name)
