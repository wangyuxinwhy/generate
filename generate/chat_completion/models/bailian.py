from __future__ import annotations

import json
import uuid
from typing import Any, AsyncIterator, ClassVar, Iterator, List, Literal, Optional

from pydantic import Field, PositiveInt
from typing_extensions import Annotated, Self, TypedDict, Unpack, override

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
from generate.platforms.bailian import BailianSettings, BailianTokenManager
from generate.types import Probability


def generate_default_request_id() -> str:
    uuid_obj = uuid.uuid4()
    return str(uuid_obj).replace('-', '')


class BailianMessage(TypedDict):
    Role: Literal['user', 'assistant', 'system']
    Content: str


def _convert_message_to_bailian_message(message: Message) -> BailianMessage:
    if isinstance(message, UserMessage):
        return {'Role': 'user', 'Content': message.content}
    if isinstance(message, AssistantMessage):
        return {'Role': 'assistant', 'Content': message.content}
    if isinstance(message, SystemMessage):
        return {'Role': 'system', 'Content': message.content}
    raise MessageTypeError(message, (UserMessage, AssistantMessage, SystemMessage))


class BailianChatParameters(ModelParameters):
    request_id: str = Field(default_factory=generate_default_request_id, alias='RequestId')
    top_p: Optional[Probability] = Field(default=None, alias='TopP')
    top_k: Optional[Annotated[int, Field(ge=0)]] = None
    seed: Optional[int] = None
    temperature: Optional[Annotated[float, Field(ge=0, le=2)]] = None
    max_tokens: Optional[PositiveInt] = None
    stop: Optional[List[str]] = None

    def custom_model_dump(self) -> dict[str, Any]:
        output = super().custom_model_dump()
        parameters = {}
        if 'top_k' in output:
            parameters['TopK'] = output.pop('top_k')
        if 'seed' in output:
            parameters['Seed'] = output.pop('seed')
        if 'temperature' in output:
            parameters['Temperature'] = output.pop('temperature')
        if 'max_tokens' in output:
            parameters['MaxTokens'] = output.pop('max_tokens')
        if 'stop' in output:
            parameters['Stop'] = output.pop('stop')
        if parameters:
            output['Parameters'] = parameters
        return output


class BailianChatParametersDict(ModelParametersDict, total=False):
    request_id: str
    top_p: Probability
    top_k: int
    seed: int
    temperature: float
    max_tokens: PositiveInt
    stop: List[str]


def calculate_bailian_cost(model_id: str, total_tokens: int) -> float | None:
    if model_id == 'qwen-turbo':
        return 0.008 * total_tokens / 1000
    if model_id == 'qwen-plus':
        return 0.012 * total_tokens / 1000
    if model_id == 'qwen-max':
        return 0.12 * total_tokens / 1000
    return None


class BailianChat(RemoteChatCompletionModel):
    model_type: ClassVar[str] = 'bailian'

    parameters: BailianChatParameters
    settings: BailianSettings

    def __init__(
        self,
        app_id: str | None = None,
        parameters: BailianChatParameters | None = None,
        settings: BailianSettings | None = None,
        http_client: HttpClient | None = None,
    ) -> None:
        parameters = parameters or BailianChatParameters()
        settings = settings or BailianSettings()  # type: ignore
        http_client = http_client or HttpClient()
        super().__init__(parameters=parameters, settings=settings, http_client=http_client)

        self.app_id = app_id or self.settings.default_app_id
        self.token_manager = BailianTokenManager(self.settings, self.http_client)

    def _get_request_parameters(self, messages: Messages, parameters: BailianChatParameters) -> HttpxPostKwargs:
        if not isinstance(messages[-1], UserMessage):
            raise MessageTypeError(messages[-1], allowed_message_type=(UserMessage,))

        json_dict = parameters.custom_model_dump()
        headers = {
            'Content-Type': 'application/json;charset=UTF-8',
            'Authorization': f'Bearer {self.token_manager.token}',
        }
        json_dict['AppId'] = self.app_id
        json_dict['Messages'] = [_convert_message_to_bailian_message(i) for i in messages]
        return {
            'url': self.settings.completion_api,
            'headers': headers,
            'json': json_dict,
        }

    @override
    def generate(self, prompt: Prompt, **kwargs: Unpack[BailianChatParametersDict]) -> ChatCompletionOutput:
        messages = ensure_messages(prompt)
        parameters = self.parameters.clone_with_changes(**kwargs)
        request_parameters = self._get_request_parameters(messages, parameters)
        response = self.http_client.post(request_parameters=request_parameters)
        return self._parse_reponse(response.json())

    @override
    async def async_generate(self, prompt: Prompt, **kwargs: Unpack[BailianChatParametersDict]) -> ChatCompletionOutput:
        messages = ensure_messages(prompt)
        parameters = self.parameters.clone_with_changes(**kwargs)
        request_parameters = self._get_request_parameters(messages, parameters)
        response = await self.http_client.async_post(request_parameters=request_parameters)
        return self._parse_reponse(response.json())

    def _parse_reponse(self, response: ResponseValue) -> ChatCompletionOutput:
        if not response['Success']:
            raise UnexpectedResponseError(response)
        response_data = response['Data']
        total_tokens = response_data['Usage'][0]['InputTokens'] + response_data['Usage'][0]['OutputTokens']
        model_id = response_data['Usage'][0]['ModelId']
        return ChatCompletionOutput(
            model_info=self.model_info,
            cost=calculate_bailian_cost(model_id=model_id, total_tokens=total_tokens),
            finish_reason=response_data.get('FinishReason'),
            message=AssistantMessage(content=response_data['Text']),
            extra={
                'thoughts': response_data.get('Thoughts'),
                'doc_references': response_data.get('DocReferences'),
                'request_id': response['RequestId'],
                'response_id': response_data.get('ResponseId'),
                'usage': response_data['Usage'][0],
            },
        )

    def _get_stream_request_parameters(self, messages: Messages, parameters: BailianChatParameters) -> HttpxPostKwargs:
        http_post_kwargs = self._get_request_parameters(messages, parameters)
        http_post_kwargs['headers']['Accept'] = 'text/event-stream'  # type: ignore
        http_post_kwargs['json']['Stream'] = True  # type: ignore
        return http_post_kwargs

    @override
    def stream_generate(
        self, prompt: Prompt, **kwargs: Unpack[BailianChatParametersDict]
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
        self, prompt: Prompt, **kwargs: Unpack[BailianChatParametersDict]
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
        response_data = parsed_line['Data']
        reply: str = response_data['Text']
        extra = {
            'thoughts': response_data.get('Thoughts'),
            'doc_references': response_data.get('DocReferences'),
            'response_id': response_data.get('ResponseId'),
        }
        if len(reply) == len(message.content):
            total_tokens = response_data['Usage'][0]['InputTokens'] + response_data['Usage'][0]['OutputTokens']
            model_id = response_data['Usage'][0]['ModelId']
            extra['usage'] = response_data['Usage'][0]
            extra['response_id'] = response_data.get('ResponseId')
            return ChatCompletionStreamOutput(
                model_info=self.model_info,
                cost=calculate_bailian_cost(model_id=model_id, total_tokens=total_tokens),
                message=message,
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
    @override
    def name(self) -> str:
        return self.app_id

    @classmethod
    @override
    def from_name(cls, name: str) -> Self:
        if name:
            raise ValueError(f'{cls} cannot be initialized from name')
        return cls()
