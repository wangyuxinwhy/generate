from __future__ import annotations

import time
from typing import Any, AsyncIterator, ClassVar, Iterator, Literal, Optional

import cachetools.func  # type: ignore
import jwt
from typing_extensions import NotRequired, Self, TypedDict, Unpack, override

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
from generate.platforms.zhipu import ZhipuSettings
from generate.types import Probability, Temperature

API_TOKEN_TTL_SECONDS = 3 * 60
CACHE_TTL_SECONDS = API_TOKEN_TTL_SECONDS - 30


class ZhipuRef(TypedDict):
    enable: NotRequired[bool]
    search_query: NotRequired[str]


class ZhipuChatParameters(ModelParameters):
    temperature: Optional[Temperature] = None
    top_p: Optional[Probability] = None
    request_id: Optional[str] = None
    search_query: Optional[str] = None

    def custom_model_dump(self) -> dict[str, Any]:
        output = super().custom_model_dump()
        if self.search_query:
            output['ref'] = {'enable': True, 'search_query': self.search_query}
        output['return_type'] = 'text'
        return output


class ZhipuChatParametersDict(ModelParametersDict, total=False):
    temperature: Optional[Temperature]
    top_p: Optional[Probability]
    request_id: Optional[str]
    search_query: Optional[str]


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

    def custom_model_dump(self) -> dict[str, Any]:
        output = super().custom_model_dump()
        output['return_type'] = 'text'
        return output


class ZhipuCharacterChatParametersDict(ModelParametersDict, total=False):
    meta: ZhipuMeta
    request_id: Optional[str]


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


class BaseZhipuChat(ChatCompletionModel):
    def __init__(
        self,
        model: str,
        parameters: ModelParameters,
        settings: ZhipuSettings | None = None,
        http_client: HttpClient | None = None,
    ) -> None:
        self.model = model
        self.parameters = parameters
        self.settings = settings or ZhipuSettings()  # type: ignore
        self.http_client = http_client or HttpClient()
        self.http_client.stream_strategy = 'basic'

    def _get_request_parameters(self, messages: Messages, parameters: ModelParameters) -> HttpxPostKwargs:
        zhipu_messages = [convert_to_zhipu_message(message) for message in messages]
        headers = {
            'Authorization': generate_token(self.settings.api_key.get_secret_value()),
        }
        params = {'prompt': zhipu_messages, **parameters.custom_model_dump()}
        return {
            'url': f'{self.settings.api_base}/{self.model}/invoke',
            'headers': headers,
            'json': params,
        }

    def _parse_reponse(self, response: ResponseValue) -> ChatCompletionOutput:
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

    def _get_stream_request_parameters(self, messages: Messages, parameters: ModelParameters) -> HttpxPostKwargs:
        http_parameters = self._get_request_parameters(messages, parameters)
        http_parameters['url'] = f'{self.settings.api_base}/{self.model}/sse-invoke'
        return http_parameters

    def calculate_cost(self, usage: dict[str, Any]) -> float | None:
        if self.name == 'chatglm_turbo':
            return 0.005 * (usage['total_tokens'] / 1000)
        if self.name == 'characterglm':
            return 0.015 * (usage['total_tokens'] / 1000)
        return None


class ZhipuChat(BaseZhipuChat):
    model_type: ClassVar[str] = 'zhipu'

    def __init__(
        self,
        model: str = 'chatglm_turbo',
        parameters: ZhipuChatParameters | None = None,
        settings: ZhipuSettings | None = None,
        http_client: HttpClient | None = None,
    ) -> None:
        parameters = parameters or ZhipuChatParameters()
        super().__init__(model=model, parameters=parameters, settings=settings, http_client=http_client)

    @override
    def generate(self, prompt: Prompt, **kwargs: Unpack[ZhipuChatParametersDict]) -> ChatCompletionOutput:
        messages = ensure_messages(prompt)
        parameters = self.parameters.update_with_validate(**kwargs)
        request_parameters = self._get_request_parameters(messages, parameters)
        response = self.http_client.post(request_parameters=request_parameters)
        return self._parse_reponse(response.json())

    @override
    async def async_generate(self, prompt: Prompt, **kwargs: Unpack[ZhipuChatParametersDict]) -> ChatCompletionOutput:
        messages = ensure_messages(prompt)
        parameters = self.parameters.update_with_validate(**kwargs)
        request_parameters = self._get_request_parameters(messages, parameters)
        response = await self.http_client.async_post(request_parameters=request_parameters)
        return self._parse_reponse(response.json())

    @override
    def stream_generate(
        self, prompt: Prompt, **kwargs: Unpack[ZhipuChatParametersDict]
    ) -> Iterator[ChatCompletionStreamOutput]:
        messages = ensure_messages(prompt)
        parameters = self.parameters.update_with_validate(**kwargs)
        request_parameters = self._get_stream_request_parameters(messages, parameters)
        yield ChatCompletionStreamOutput(
            model_info=self.model_info,
            stream=Stream(delta='', control='start'),
        )
        message = ''
        for line in self.http_client.stream_post(request_parameters=request_parameters):
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
    async def async_stream_generate(
        self, prompt: Prompt, **kwargs: Unpack[ZhipuChatParametersDict]
    ) -> AsyncIterator[ChatCompletionStreamOutput]:
        messages = ensure_messages(prompt)
        parameters = self.parameters.update_with_validate(**kwargs)
        request_parameters = self._get_stream_request_parameters(messages, parameters)
        yield ChatCompletionStreamOutput(
            model_info=self.model_info,
            stream=Stream(delta='', control='start'),
        )
        message = ''
        async for line in self.http_client.async_stream_post(request_parameters=request_parameters):
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

    @property
    @override
    def name(self) -> str:
        return self.model

    @classmethod
    @override
    def from_name(cls, name: str) -> Self:
        return cls(model=name)


class ZhipuCharacterChat(BaseZhipuChat):
    model_type: ClassVar[str] = 'zhipu-character'

    def __init__(
        self,
        model: str = 'characterglm',
        parameters: ZhipuCharacterChatParameters | None = None,
        settings: ZhipuSettings | None = None,
        http_client: HttpClient | None = None,
    ) -> None:
        parameters = parameters or ZhipuCharacterChatParameters()
        super().__init__(model=model, parameters=parameters, settings=settings, http_client=http_client)

    @override
    def generate(self, prompt: Prompt, **kwargs: Unpack[ZhipuCharacterChatParametersDict]) -> ChatCompletionOutput:
        messages = ensure_messages(prompt)
        parameters = self.parameters.update_with_validate(**kwargs)
        request_parameters = self._get_request_parameters(messages, parameters)
        response = self.http_client.post(request_parameters=request_parameters)
        return self._parse_reponse(response.json())

    @override
    async def async_generate(self, prompt: Prompt, **kwargs: Unpack[ZhipuCharacterChatParametersDict]) -> ChatCompletionOutput:
        messages = ensure_messages(prompt)
        parameters = self.parameters.update_with_validate(**kwargs)
        request_parameters = self._get_request_parameters(messages, parameters)
        response = await self.http_client.async_post(request_parameters=request_parameters)
        return self._parse_reponse(response.json())

    @override
    def stream_generate(
        self, prompt: Prompt, **kwargs: Unpack[ZhipuCharacterChatParametersDict]
    ) -> Iterator[ChatCompletionStreamOutput]:
        messages = ensure_messages(prompt)
        parameters = self.parameters.update_with_validate(**kwargs)
        request_parameters = self._get_stream_request_parameters(messages, parameters)
        yield ChatCompletionStreamOutput(
            model_info=self.model_info,
            stream=Stream(delta='', control='start'),
        )
        message = ''
        for line in self.http_client.stream_post(request_parameters=request_parameters):
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
    async def async_stream_generate(
        self, prompt: Prompt, **kwargs: Unpack[ZhipuCharacterChatParametersDict]
    ) -> AsyncIterator[ChatCompletionStreamOutput]:
        messages = ensure_messages(prompt)
        parameters = self.parameters.update_with_validate(**kwargs)
        request_parameters = self._get_stream_request_parameters(messages, parameters)
        yield ChatCompletionStreamOutput(
            model_info=self.model_info,
            stream=Stream(delta='', control='start'),
        )
        message = ''
        async for line in self.http_client.async_stream_post(request_parameters=request_parameters):
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

    @property
    @override
    def name(self) -> str:
        return self.model

    @classmethod
    @override
    def from_name(cls, name: str) -> Self:
        return cls(model=name)
