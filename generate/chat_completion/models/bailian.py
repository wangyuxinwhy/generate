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
    MessageTypeError,
    Prompt,
    SystemMessage,
    UserMessage,
    ensure_messages,
)
from generate.chat_completion.model_output import ChatCompletionOutput, ChatCompletionStreamOutput
from generate.chat_completion.stream_manager import StreamManager
from generate.http import (
    HttpClient,
    HttpxPostKwargs,
    ResponseValue,
    UnexpectedResponseError,
)
from generate.model import ModelParameters, RemoteModelParametersDict
from generate.platforms.bailian import BailianSettings, BailianTokenManager
from generate.types import Probability


def generate_default_request_id() -> str:
    uuid_obj = uuid.uuid4()
    return str(uuid_obj).replace('-', '')


class BailianMessage(TypedDict):
    Role: Literal['user', 'assistant', 'system']
    Content: str


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


class BailianChatParametersDict(RemoteModelParametersDict, total=False):
    request_id: str
    top_p: Optional[Probability]
    top_k: Optional[int]
    seed: Optional[int]
    temperature: Optional[float]
    max_tokens: Optional[PositiveInt]
    stop: Optional[List[str]]


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
        self.app_id = app_id or settings.default_app_id
        super().__init__(model=self.app_id, parameters=parameters, settings=settings, http_client=http_client)

        self.token_manager = BailianTokenManager(self.settings, self.http_client)

    @override
    def generate(self, prompt: Prompt, **kwargs: Unpack[BailianChatParametersDict]) -> ChatCompletionOutput:
        return super().generate(prompt, **kwargs)

    @override
    async def async_generate(self, prompt: Prompt, **kwargs: Unpack[BailianChatParametersDict]) -> ChatCompletionOutput:
        return await super().async_generate(prompt, **kwargs)

    @override
    def stream_generate(
        self, prompt: Prompt, **kwargs: Unpack[BailianChatParametersDict]
    ) -> Iterator[ChatCompletionStreamOutput]:
        yield from super().stream_generate(prompt, **kwargs)

    @override
    async def async_stream_generate(
        self, prompt: Prompt, **kwargs: Unpack[BailianChatParametersDict]
    ) -> AsyncIterator[ChatCompletionStreamOutput]:
        async for output in super().async_stream_generate(prompt, **kwargs):
            yield output

    @classmethod
    @override
    def from_name(cls, name: str) -> Self:
        return cls(app_id=name)

    @staticmethod
    def _convert_message(message: Message) -> BailianMessage:
        if isinstance(message, UserMessage):
            return {'Role': 'user', 'Content': message.content}
        if isinstance(message, AssistantMessage):
            return {'Role': 'assistant', 'Content': message.content}
        if isinstance(message, SystemMessage):
            return {'Role': 'system', 'Content': message.content}
        raise MessageTypeError(message, (UserMessage, AssistantMessage, SystemMessage))

    @override
    def _get_request_parameters(
        self, prompt: Prompt, stream: bool = False, **kwargs: Unpack[BailianChatParametersDict]
    ) -> HttpxPostKwargs:
        messages = ensure_messages(prompt)
        parameters = self.parameters.clone_with_changes(**kwargs)

        if not isinstance(messages[-1], UserMessage):
            raise MessageTypeError(messages[-1], allowed_message_type=(UserMessage,))

        json_dict = parameters.custom_model_dump()
        headers = {
            'Content-Type': 'application/json;charset=UTF-8',
            'Authorization': f'Bearer {self.token_manager.token}',
        }
        json_dict['AppId'] = self.app_id
        json_dict['Messages'] = [self._convert_message(i) for i in messages]

        if stream:
            headers['Accept'] = 'text/event-stream'
            json_dict['Stream'] = True

        return {
            'url': self.settings.completion_api,
            'headers': headers,
            'json': json_dict,
        }

    @override
    def _process_reponse(self, response: ResponseValue) -> ChatCompletionOutput:
        if not response['Success']:
            raise UnexpectedResponseError(response)
        response_data = response['Data']
        total_tokens = response_data['Usage'][0]['InputTokens'] + response_data['Usage'][0]['OutputTokens']
        model_id = response_data['Usage'][0]['ModelId']
        return ChatCompletionOutput(
            model_info=self.model_info,
            cost=self._calculate_cost(model_id=model_id, total_tokens=total_tokens),
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

    @override
    def _process_stream_line(self, line: str, stream_manager: StreamManager) -> ChatCompletionStreamOutput | None:
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            return None

        response_data = data['Data']
        reply: str = response_data['Text']
        stream_manager.extra.update(
            {
                'thoughts': response_data.get('Thoughts'),
                'doc_references': response_data.get('DocReferences'),
                'response_id': response_data.get('ResponseId'),
            }
        )

        is_finish = len(reply) == len(stream_manager.content)
        if is_finish:
            total_tokens = response_data['Usage'][0]['InputTokens'] + response_data['Usage'][0]['OutputTokens']
            model_id = response_data['Usage'][0]['ModelId']
            stream_manager.delta = ''
            stream_manager.extra['usage'] = response_data['Usage'][0]
            stream_manager.extra['response_id'] = response_data.get('ResponseId')
            stream_manager.cost = self._calculate_cost(model_id=model_id, total_tokens=total_tokens)
            stream_manager.finish_reason = 'stop'
            return stream_manager.build_stream_output()

        delta = reply[len(stream_manager.content) :]
        stream_manager.delta = delta
        return stream_manager.build_stream_output()

    @staticmethod
    def _calculate_cost(model_id: str, total_tokens: int) -> float | None:
        if model_id == 'qwen-turbo':
            return 0.008 * total_tokens / 1000
        if model_id == 'qwen-plus':
            return 0.012 * total_tokens / 1000
        if model_id == 'qwen-max':
            return 0.12 * total_tokens / 1000
        return None
