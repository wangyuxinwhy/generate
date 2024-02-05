from __future__ import annotations

import hashlib
import json
from io import BytesIO
from typing import AsyncIterator, ClassVar, Iterator, List, Literal, Optional

from pydantic import Field, PositiveInt
from typing_extensions import Annotated, Self, TypedDict, Unpack, override

from generate.chat_completion.base import RemoteChatCompletionModel
from generate.chat_completion.message import (
    AssistantMessage,
    ImagePart,
    ImageUrlPart,
    Message,
    Messages,
    MessageTypeError,
    Prompt,
    SystemMessage,
    TextPart,
    UserMessage,
    UserMultiPartMessage,
    ensure_messages,
)
from generate.chat_completion.model_output import ChatCompletionOutput, ChatCompletionStreamOutput, Stream
from generate.http import (
    HttpClient,
    HttpGetKwargs,
    HttpxPostKwargs,
    ResponseValue,
)
from generate.model import ModelParameters, ModelParametersDict
from generate.platforms.dashscope import DashScopeSettings
from generate.types import Probability


class DashScopeChatParameters(ModelParameters):
    seed: Optional[PositiveInt] = None
    max_tokens: Optional[PositiveInt] = None
    top_p: Optional[Probability] = Field(default=None, alias='TopP')
    top_k: Optional[Annotated[int, Field(ge=0, le=100)]] = None
    repetition_penalty: Optional[float] = None
    temperature: Optional[Annotated[float, Field(gt=0, le=2)]] = None
    stop: Optional[List[str]] = None
    enable_search: Optional[bool] = None


class DashScopeChatParametersDict(ModelParametersDict, total=False):
    seed: int
    max_tokens: int
    top_p: float
    top_k: int
    repetition_penalty: float
    temperature: float
    stop: List[str]
    enable_search: bool


def _convert_message_to_chat_message(message: Message) -> dict[str, str]:
    if isinstance(message, UserMessage):
        return {'role': 'user', 'content': message.content}
    if isinstance(message, AssistantMessage):
        return {'role': 'assistant', 'content': message.content}
    if isinstance(message, SystemMessage):
        return {'role': 'system', 'content': message.content}
    raise MessageTypeError(message, (UserMessage, AssistantMessage, SystemMessage))


def _calculate_cost(model_name: str, total_tokens: int) -> Optional[float]:
    return None


class DashScopeChat(RemoteChatCompletionModel):
    model_type: ClassVar[str] = 'dashscope'

    parameters: DashScopeChatParameters
    settings: DashScopeSettings

    def __init__(
        self,
        model: str = 'qwen-max',
        parameters: DashScopeChatParameters | None = None,
        settings: DashScopeSettings | None = None,
        http_client: HttpClient | None = None,
    ) -> None:
        parameters = parameters or DashScopeChatParameters()
        settings = settings or DashScopeSettings()  # type: ignore
        http_client = http_client or HttpClient()
        super().__init__(parameters=parameters, settings=settings, http_client=http_client)

        self.model = model

    def _get_request_parameters(self, messages: Messages, parameters: DashScopeChatParameters) -> HttpxPostKwargs:
        zhipu_messages = [_convert_message_to_chat_message(message) for message in messages]
        headers = {
            'Authorization': self.settings.api_key.get_secret_value(),
            'Content-Type': 'application/json',
        }
        params = {
            'input': {
                'messages': zhipu_messages,
            },
            'model': self.model,
            'parameters': parameters.custom_model_dump(),
        }
        return {
            'url': f'{self.settings.api_base}/services/aigc/text-generation/generation',
            'headers': headers,
            'json': params,
        }

    @override
    def generate(self, prompt: Prompt, **kwargs: Unpack[DashScopeChatParametersDict]) -> ChatCompletionOutput:
        messages = ensure_messages(prompt)
        parameters = self.parameters.clone_with_changes(**kwargs)
        request_parameters = self._get_request_parameters(messages, parameters)
        response = self.http_client.post(request_parameters=request_parameters)
        return self._parse_reponse(response.json())

    @override
    async def async_generate(self, prompt: Prompt, **kwargs: Unpack[DashScopeChatParametersDict]) -> ChatCompletionOutput:
        messages = ensure_messages(prompt)
        parameters = self.parameters.clone_with_changes(**kwargs)
        request_parameters = self._get_request_parameters(messages, parameters)
        response = await self.http_client.async_post(request_parameters=request_parameters)
        return self._parse_reponse(response.json())

    def _parse_reponse(self, response: ResponseValue) -> ChatCompletionOutput:
        return ChatCompletionOutput(
            model_info=self.model_info,
            message=AssistantMessage(content=response['output']['text']),
            cost=_calculate_cost(self.model, response['usage']['total_tokens']),
            extra={'usage': response['usage'], 'request_id': response['request_id']},
        )

    def _get_stream_request_parameters(self, messages: Messages, parameters: DashScopeChatParameters) -> HttpxPostKwargs:
        http_parameters = self._get_request_parameters(messages, parameters)
        http_parameters['headers']['Accept'] = 'text/event-stream'
        return http_parameters

    @override
    def stream_generate(
        self, prompt: Prompt, **kwargs: Unpack[DashScopeChatParametersDict]
    ) -> Iterator[ChatCompletionStreamOutput]:
        messages = ensure_messages(prompt)
        parameters = self.parameters.clone_with_changes(**kwargs)
        request_parameters = self._get_stream_request_parameters(messages, parameters)
        message = AssistantMessage(content='')
        is_start = True
        is_finish = False
        for line in self.http_client.stream_post(request_parameters=request_parameters):
            if is_finish:
                continue

            output = self._parse_stream_line(line, message, is_start)
            is_start = False
            is_finish = output.is_finish
            yield output

    @override
    async def async_stream_generate(
        self, prompt: Prompt, **kwargs: Unpack[DashScopeChatParametersDict]
    ) -> AsyncIterator[ChatCompletionStreamOutput]:
        messages = ensure_messages(prompt)
        parameters = self.parameters.clone_with_changes(**kwargs)
        request_parameters = self._get_stream_request_parameters(messages, parameters)
        message = AssistantMessage(content='')
        is_start = True
        is_finish = False
        async for line in self.http_client.async_stream_post(request_parameters=request_parameters):
            if is_finish:
                continue

            output = self._parse_stream_line(line, message, is_start)
            is_start = False
            is_finish = output.is_finish
            yield output

    def _parse_stream_line(self, line: str, message: AssistantMessage, is_start: bool) -> ChatCompletionStreamOutput:
        parsed_line = json.loads(line)
        finish_reason = parsed_line['output']['finish_reason']
        reply = parsed_line['output']['text']
        usage = parsed_line['usage']
        request_id = parsed_line['request_id']
        extra = {
            'usage': usage,
            'response_id': request_id,
        }
        if finish_reason == 'stop':
            return ChatCompletionStreamOutput(
                model_info=self.model_info,
                cost=_calculate_cost(self.model, total_tokens=usage['total_tokens']),
                message=AssistantMessage(content=reply),
                extra=extra,
                finish_reason='stop',
                stream=Stream(delta='', control='finish'),
            )
        delta = reply[len(message.content) :]
        message.content = reply
        return ChatCompletionStreamOutput(
            model_info=self.model_info,
            message=message,
            extra=extra,
            stream=Stream(delta=delta, control='start' if is_start else 'continue'),
        )

    @property
    def name(self) -> str:
        return self.model

    @classmethod
    def from_name(cls, name: str) -> Self:
        return cls(model=name)


class DashScopeMultiModalChatParameters(ModelParameters):
    seed: Optional[PositiveInt] = None
    top_p: Optional[Probability] = Field(default=None, alias='TopP')
    top_k: Optional[Annotated[int, Field(ge=0, le=100)]] = None


class DashScopeMultiModalChatParametersDict(ModelParametersDict, total=False):
    seed: int
    top_p: float
    top_k: int


class DashScopeMultiModalMessage(TypedDict):
    role: str
    content: list[dict[Literal['image', 'text'], str]]


class DashScopeMultiModalChat(RemoteChatCompletionModel):
    model_type: ClassVar[str] = 'dashscope_multimodal'

    parameters: DashScopeMultiModalChatParameters
    settings: DashScopeSettings

    def __init__(
        self,
        model: str = 'qwen-vl-max',
        parameters: DashScopeMultiModalChatParameters | None = None,
        settings: DashScopeSettings | None = None,
        http_client: HttpClient | None = None,
    ) -> None:
        parameters = parameters or DashScopeMultiModalChatParameters()
        settings = settings or DashScopeSettings()  # type: ignore
        http_client = http_client or HttpClient()
        super().__init__(parameters=parameters, settings=settings, http_client=http_client)

        self.model = model

    def _get_request_parameters(self, messages: Messages, parameters: DashScopeMultiModalChatParameters) -> HttpxPostKwargs:
        dashscope_messages = [self.convert_message_to_multimodal_chat_message(message) for message in messages]
        headers = {
            'Authorization': self.settings.api_key.get_secret_value(),
            'Content-Type': 'application/json',
        }
        if any(self.has_oss_file(message) for message in dashscope_messages):
            headers['X-DashScope-OssResourceResolve'] = 'enable'
        params = {
            'input': {
                'messages': dashscope_messages,
            },
            'model': self.model,
            'parameters': parameters.custom_model_dump(),
        }
        return {
            'url': f'{self.settings.api_base}/services/aigc/multimodal-generation/generation',
            'headers': headers,
            'json': params,
        }

    def has_oss_file(self, message: DashScopeMultiModalMessage) -> bool:
        for part in message['content']:
            for k, v in part.items():
                if k == 'image' and v.startswith('oss://'):
                    return True
        return False

    def convert_message_to_multimodal_chat_message(self, message: Message) -> DashScopeMultiModalMessage:
        if isinstance(message, UserMessage):
            return {'role': 'user', 'content': [{'text': message.content}]}
        if isinstance(message, AssistantMessage):
            return {'role': 'assistant', 'content': [{'text': message.content}]}
        if isinstance(message, SystemMessage):
            return {'role': 'system', 'content': [{'text': message.content}]}
        if isinstance(message, UserMultiPartMessage):
            content = []
            for part in message.content:
                if isinstance(part, TextPart):
                    content.append({'text': part.text})
                elif isinstance(part, ImageUrlPart):
                    content.append({'image': part.image_url.url})
                elif isinstance(part, ImagePart):
                    image_url = self.upload_image(part.image, part.image_format or 'png')
                    content.append({'image': image_url})
                else:
                    raise TypeError(f'Unsupported part type: {part}')
            return {'role': 'user', 'content': content}
        allowed_message_type = (UserMessage, AssistantMessage, SystemMessage, UserMultiPartMessage)
        raise MessageTypeError(message, allowed_message_type=allowed_message_type)

    def upload_image(self, image: bytes, image_format: str) -> str:
        get_kwargs = HttpGetKwargs(
            url=f'{self.settings.api_base}/uploads',
            params={'action': 'getPolicy', 'model': self.model},
            headers={'Authorization': f'Bearer {self.settings.api_key.get_secret_value()}'},
        )
        response_data = self.http_client.get(get_kwargs)
        upload_info = response_data.json()['data']

        form_data = {}
        form_data['OSSAccessKeyId'] = upload_info['oss_access_key_id']
        form_data['Signature'] = upload_info['signature']
        form_data['policy'] = upload_info['policy']
        hash_code = hashlib.md5(image).hexdigest()
        form_data['key'] = upload_info['upload_dir'] + '/' + f'{hash_code}.{image_format}'
        form_data['x-oss-object-acl'] = upload_info['x_oss_object_acl']
        form_data['x-oss-forbid-overwrite'] = upload_info['x_oss_forbid_overwrite']
        form_data['success_action_status'] = '200'
        form_data['x-oss-content-type'] = f'image/{image_format}'
        url = upload_info['upload_host']
        files = {'file': BytesIO(image)}
        response = self.http_client.client.post(
            url,
            data=form_data,
            files=files,
        )
        response.raise_for_status()
        return 'oss://' + form_data['key']

    def _get_stream_request_parameters(
        self, messages: Messages, parameters: DashScopeMultiModalChatParameters
    ) -> HttpxPostKwargs:
        http_parameters = self._get_request_parameters(messages, parameters)
        http_parameters['headers']['Accept'] = 'text/event-stream'
        return http_parameters

    def _parse_reponse(self, response: ResponseValue) -> ChatCompletionOutput:
        choice = response['output']['choices'][0]
        return ChatCompletionOutput(
            model_info=self.model_info,
            message=AssistantMessage(content=choice['message']['content'][0]['text']),
            cost=None,
            extra={'usage': response['usage'], 'request_id': response['request_id']},
        )

    @override
    def generate(self, prompt: Prompt, **kwargs: Unpack[DashScopeMultiModalChatParametersDict]) -> ChatCompletionOutput:
        messages = ensure_messages(prompt)
        parameters = self.parameters.clone_with_changes(**kwargs)
        request_parameters = self._get_request_parameters(messages, parameters)
        response = self.http_client.post(request_parameters=request_parameters)
        return self._parse_reponse(response.json())

    @override
    async def async_generate(
        self, prompt: Prompt, **kwargs: Unpack[DashScopeMultiModalChatParametersDict]
    ) -> ChatCompletionOutput:
        messages = ensure_messages(prompt)
        parameters = self.parameters.clone_with_changes(**kwargs)
        request_parameters = self._get_request_parameters(messages, parameters)
        response = await self.http_client.async_post(request_parameters=request_parameters)
        return self._parse_reponse(response.json())

    @override
    def stream_generate(
        self, prompt: Prompt, **kwargs: Unpack[DashScopeMultiModalChatParametersDict]
    ) -> Iterator[ChatCompletionStreamOutput]:
        messages = ensure_messages(prompt)
        parameters = self.parameters.clone_with_changes(**kwargs)
        request_parameters = self._get_stream_request_parameters(messages, parameters)
        message = AssistantMessage(content='')
        is_start = True
        is_finish = False
        for line in self.http_client.stream_post(request_parameters=request_parameters):
            if is_finish:
                continue

            output = self._parse_stream_line(line, message, is_start)
            is_start = False
            is_finish = output.is_finish
            yield output

    @override
    async def async_stream_generate(
        self, prompt: Prompt, **kwargs: Unpack[DashScopeMultiModalChatParametersDict]
    ) -> AsyncIterator[ChatCompletionStreamOutput]:
        messages = ensure_messages(prompt)
        parameters = self.parameters.clone_with_changes(**kwargs)
        request_parameters = self._get_stream_request_parameters(messages, parameters)
        message = AssistantMessage(content='')
        is_start = True
        is_finish = False
        async for line in self.http_client.async_stream_post(request_parameters=request_parameters):
            if is_finish:
                continue

            output = self._parse_stream_line(line, message, is_start)
            is_start = False
            is_finish = output.is_finish
            yield output

    def _parse_stream_line(self, line: str, message: AssistantMessage, is_start: bool) -> ChatCompletionStreamOutput:
        parsed_line = json.loads(line)
        choice = parsed_line['output']['choices'][0]
        finish_reason = choice['finish_reason']
        reply = choice['message']['content'][0]['text']
        usage = parsed_line['usage']
        request_id = parsed_line['request_id']
        extra = {
            'usage': usage,
            'response_id': request_id,
        }
        if finish_reason == 'stop':
            return ChatCompletionStreamOutput(
                model_info=self.model_info,
                cost=None,
                message=AssistantMessage(content=reply),
                extra=extra,
                finish_reason='stop',
                stream=Stream(delta='', control='finish'),
            )
        delta = reply[len(message.content) :]
        message.content = reply
        return ChatCompletionStreamOutput(
            model_info=self.model_info,
            message=message,
            extra=extra,
            stream=Stream(delta=delta, control='start' if is_start else 'continue'),
        )

    @property
    def name(self) -> str:
        return self.model

    @classmethod
    def from_name(cls, name: str) -> Self:
        return cls(model=name)
