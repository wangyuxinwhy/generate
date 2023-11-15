from __future__ import annotations

import os
import time
from typing import Any, AsyncIterator, ClassVar, Iterator, Literal, Optional, TypeVar

import cachetools.func  # type: ignore
import jwt
from typing_extensions import NotRequired, Self, TypedDict, Unpack, override

from generate.chat_completion.base import ChatCompletionModel
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

P = TypeVar('P', bound=ModelParameters)
API_TOKEN_TTL_SECONDS = 3 * 60
CACHE_TTL_SECONDS = API_TOKEN_TTL_SECONDS - 30


class ZhipuRef(TypedDict):
    enable: NotRequired[bool]
    search_query: NotRequired[str]


class ZhipuChatParameters(ModelParameters):
    temperature: Optional[Temperature] = None
    top_p: Optional[Probability] = None
    request_id: Optional[str] = None
    ref: Optional[ZhipuRef] = None


class ZhipuMeta(TypedDict):
    user_info: str
    bot_info: str
    bot_name: str
    user_name: str


class ZhipuCharacterChatParameters(ModelParameters):
    meta: ZhipuMeta = {
        'user_info': '我是陆星辰，是一个男性，是一位知名导演，也是苏梦远的合作导演。',
        'bot_info': '苏梦远，本名苏远心，是一位当红的国内女歌手及演员。',
        'bot_name': '苏梦远',
        'user_name': '陆星辰',
    }
    request_id: Optional[str] = None


class ZhipuMessage(TypedDict):
    role: Literal['user', 'assistant']
    content: str


def convert_to_zhipu_message(message: Message) -> ZhipuMessage:
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


@cachetools.func.ttl_cache(maxsize=10, ttl=CACHE_TTL_SECONDS)
def generate_token(api_key: str) -> str:
    try:
        api_key, secret = api_key.split('.')
    except Exception as e:
        raise ValueError('invalid api_key') from e

    payload = {
        'api_key': api_key,
        'exp': int(round(time.time() * 1000)) + API_TOKEN_TTL_SECONDS * 1000,
        'timestamp': int(round(time.time() * 1000)),
    }

    return jwt.encode(  # type: ignore
        payload,
        secret,
        algorithm='HS256',
        headers={'alg': 'HS256', 'sign_type': 'SIGN'},
    )


class BaseZhipuChat(ChatCompletionModel[P], HttpMixin):
    default_api_base: ClassVar[str] = 'https://open.bigmodel.cn/api/paas/v3/model-api'

    def __init__(
        self,
        model: str,
        parameters: P,
        api_key: str | None = None,
        api_base: str | None = None,
        **kwargs: Unpack[HttpClientInitKwargs],
    ) -> None:
        super().__init__(parameters=parameters)
        self.model = model
        self.api_key = api_key or os.environ['ZHIPU_API_KEY']
        self.api_base = (api_base or self.default_api_base).rstrip('/')
        self.http_client = HttpClient(**kwargs)
        self.http_stream_client = HttpStreamClient(stream_strategy='basic', **kwargs)

    def _get_request_parameters(self, messages: Messages, parameters: P) -> HttpxPostKwargs:
        zhipu_messages = [convert_to_zhipu_message(message) for message in messages]
        headers = {
            'Authorization': generate_token(self.api_key),
        }
        parameters_dict = parameters.model_dump(exclude_none=True)
        params = {'prompt': zhipu_messages, **parameters_dict}
        return {
            'url': f'{self.api_base}/{self.model}/invoke',
            'headers': headers,
            'json': params,
        }

    @override
    def _completion(self, messages: Messages, parameters: P) -> ChatCompletionOutput:
        request_parameters = self._get_request_parameters(messages, parameters)
        response = self.http_client.post(request_parameters=request_parameters)
        return self._parse_reponse(response.json())

    @override
    async def _async_completion(self, messages: Messages, parameters: P) -> ChatCompletionOutput:
        request_parameters = self._get_request_parameters(messages, parameters)
        response = await self.http_client.async_post(request_parameters=request_parameters)
        return self._parse_reponse(response.json())

    def _parse_reponse(self, response: dict[str, Any]) -> ChatCompletionOutput:
        if response['success']:
            text = response['data']['choices'][0]['content']
            messages = [AssistantMessage(content=text)]
            return ChatCompletionOutput(
                model_info=self.model_info,
                messages=messages,
                cost=self.calculate_cost(response['data']['usage']),
                extra={'usage': response['data']['usage']},
            )

        raise UnexpectedResponseError(response)

    def _get_stream_request_parameters(self, messages: Messages, parameters: P) -> HttpxPostKwargs:
        http_parameters = self._get_request_parameters(messages, parameters)
        http_parameters['url'] = f'{self.api_base}/{self.model}/sse-invoke'
        return http_parameters

    @override
    def _stream_completion(self, messages: Messages, parameters: P) -> Iterator[ChatCompletionStreamOutput]:
        request_parameters = self._get_stream_request_parameters(messages, parameters)
        yield ChatCompletionStreamOutput(
            model_info=self.model_info,
            stream=Stream(delta='', control='start'),
        )
        message = ''
        for line in self.http_stream_client.post(request_parameters=request_parameters):
            message += line
            yield ChatCompletionStreamOutput(
                model_info=self.model_info,
                stream=Stream(delta=line, control='continue'),
            )
        yield ChatCompletionStreamOutput(
            model_info=self.model_info,
            messages=[AssistantMessage(content=message)],
            stream=Stream(delta='', control='finish'),
        )

    @override
    async def _async_stream_completion(self, messages: Messages, parameters: P) -> AsyncIterator[ChatCompletionStreamOutput]:
        request_parameters = self._get_stream_request_parameters(messages, parameters)
        yield ChatCompletionStreamOutput(
            model_info=self.model_info,
            stream=Stream(delta='', control='start'),
        )
        message = ''
        async for line in self.http_stream_client.async_post(request_parameters=request_parameters):
            message += line
            yield ChatCompletionStreamOutput(
                model_info=self.model_info,
                stream=Stream(delta=line, control='continue'),
            )
        yield ChatCompletionStreamOutput(
            model_info=self.model_info,
            messages=[AssistantMessage(content=message)],
            stream=Stream(delta='', control='finish'),
        )

    def calculate_cost(self, usage: dict[str, Any]) -> float | None:
        if self.name == 'chatglm_turbo':
            return 0.005 * (usage['total_tokens'] / 1000)
        if self.name == 'characterglm':
            return 0.015 * (usage['total_tokens'] / 1000)
        return None

    @property
    @override
    def name(self) -> str:
        return self.model

    @classmethod
    @override
    def from_name(cls, name: str, **kwargs: Any) -> Self:
        return cls(model=name, **kwargs)


class ZhipuChat(BaseZhipuChat[ZhipuChatParameters]):
    model_type: ClassVar[str] = 'zhipu'

    def __init__(
        self,
        model: str = 'chatglm_turbo',
        api_key: str | None = None,
        api_base: str | None = None,
        parameters: ZhipuChatParameters | None = None,
        **kwargs: Unpack[HttpClientInitKwargs],
    ) -> None:
        parameters = parameters or ZhipuChatParameters()
        super().__init__(model=model, api_key=api_key, api_base=api_base, parameters=parameters, **kwargs)


class ZhipuCharacterChat(BaseZhipuChat[ZhipuCharacterChatParameters]):
    model_type: ClassVar[str] = 'zhipu-character'

    def __init__(
        self,
        model: str = 'characterglm',
        api_key: str | None = None,
        api_base: str | None = None,
        parameters: ZhipuCharacterChatParameters | None = None,
        **kwargs: Unpack[HttpClientInitKwargs],
    ) -> None:
        parameters = parameters or ZhipuCharacterChatParameters()
        super().__init__(model=model, api_key=api_key, api_base=api_base, parameters=parameters, **kwargs)
