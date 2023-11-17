from __future__ import annotations

import json
import uuid
from typing import Any, AsyncIterator, ClassVar, Iterator, List, Literal, Optional

from pydantic import Field
from typing_extensions import Annotated, NotRequired, Self, TypedDict, override

from generate.chat_completion.base import ChatCompletionModel
from generate.chat_completion.message import (
    AssistantMessage,
    Messages,
    MessageTypeError,
    UserMessage,
)
from generate.chat_completion.model_output import ChatCompletionOutput, ChatCompletionStreamOutput, Stream
from generate.http import (
    HttpClient,
    HttpxPostKwargs,
    UnexpectedResponseError,
)
from generate.model import ModelParameters
from generate.platforms.bailian import BailianSettings
from generate.token import TokenMixin
from generate.types import Probability


class BailianChatQAPair(TypedDict):
    User: str
    Bot: str


class BailianParameter(TypedDict):
    TokK: NotRequired[int]
    Seed: NotRequired[int]
    UseRawPrompt: NotRequired[bool]


def generate_default_request_id() -> str:
    uuid_obj = uuid.uuid4()
    return str(uuid_obj).replace('-', '')


def convert_to_bailian_chat_qa_pair(messages: Messages) -> list[BailianChatQAPair]:
    pairs: list[BailianChatQAPair] = []
    for user_message, assistant_message in zip(messages[::2], messages[1::2]):
        if not isinstance(user_message, UserMessage):
            raise MessageTypeError(user_message, allowed_message_type=(UserMessage,))
        if not isinstance(assistant_message, AssistantMessage):
            raise MessageTypeError(assistant_message, allowed_message_type=(AssistantMessage,))
        pairs.append({'User': user_message.content, 'Bot': assistant_message.content})
    return pairs


class BailianChatParameters(ModelParameters):
    request_id: str = Field(default_factory=generate_default_request_id, alias='RequestId')
    session_id: Optional[str] = Field(default=None, alias='SessionId')
    top_p: Optional[Probability] = Field(default=None, alias='TopP')
    has_thoughts: Optional[bool] = Field(default=None, alias='HasThoughts')
    doc_reference_type: Literal['indexed', 'simpole'] = Field(default=None, alias='DocReferenceType')
    top_k: Optional[Annotated[int, Field(ge=0)]] = None
    seed: Optional[int] = None
    use_raw_prompt: Optional[bool] = None
    doc_tag_ids: Optional[List[int]] = Field(default=None, alias='DocTagIds')

    def custom_model_dump(self) -> dict[str, Any]:
        output = super().custom_model_dump()
        parameters = {}
        if 'top_k' in output:
            parameters['TopK'] = output.pop('top_k')
        if 'seed' in output:
            parameters['Seed'] = output.pop('seed')
        if 'use_raw_prompt' in output:
            parameters['UseRawPrompt'] = output.pop('use_raw_prompt')
        if parameters:
            output['Parameters'] = parameters
        return output


class BailianChat(ChatCompletionModel[BailianChatParameters], TokenMixin):
    model_type: ClassVar[str] = 'bailian'

    def __init__(
        self,
        settings: BailianSettings | None = None,
        parameters: BailianChatParameters | None = None,
        http_client: HttpClient | None = None,
    ) -> None:
        parameters = parameters or BailianChatParameters()
        super().__init__(parameters=parameters)

        self._token = None
        self.settings = settings or BailianSettings()  # type: ignore
        self.http_client = http_client or HttpClient()

    def _get_token(self) -> str:
        try:
            import broadscope_bailian
        except ImportError as e:
            raise ImportError('Please install broadscope_bailian first: pip install broadscope_bailian') from e

        client = broadscope_bailian.AccessTokenClient(
            access_key_id=self.settings.access_key_id.get_secret_value(),
            access_key_secret=self.settings.access_key_secret.get_secret_value(),
            agent_key=self.settings.agent_key,
        )
        return client.get_token()

    def _get_request_parameters(self, messages: Messages, parameters: BailianChatParameters) -> HttpxPostKwargs:
        if not isinstance(messages[-1], UserMessage):
            raise MessageTypeError(messages[-1], allowed_message_type=(UserMessage,))

        prompt = messages[-1].content
        history = convert_to_bailian_chat_qa_pair(messages[:-1])

        json_dict = parameters.custom_model_dump()
        headers = {
            'Content-Type': 'application/json;charset=UTF-8',
            'Authorization': f'Bearer {self.token}',
        }
        json_dict['Prompt'] = prompt
        json_dict['AppId'] = self.settings.app_id
        json_dict['History'] = history
        return {
            'url': self.settings.completion_api,
            'headers': headers,
            'json': json_dict,
        }

    @override
    def _completion(self, messages: Messages, parameters: BailianChatParameters) -> ChatCompletionOutput:
        request_parameters = self._get_request_parameters(messages, parameters)
        response = self.http_client.post(request_parameters=request_parameters)
        return self._parse_reponse(response.json())

    @override
    async def _async_completion(self, messages: Messages, parameters: BailianChatParameters) -> ChatCompletionOutput:
        request_parameters = self._get_request_parameters(messages, parameters)
        response = await self.http_client.async_post(request_parameters=request_parameters)
        return self._parse_reponse(response.json())

    def _parse_reponse(self, response: dict[str, Any]) -> ChatCompletionOutput:
        if not response['Success']:
            raise UnexpectedResponseError(response)
        messages = [AssistantMessage(content=response['Data']['Text'])]
        return ChatCompletionOutput(
            model_info=self.model_info,
            messages=messages,
            extra={
                'thoughts': response['Data']['Thoughts'],
                'doc_references': response['Data']['DocReferences'],
                'request_id': response['RequestId'],
                'response_id': response['Data']['ResponseId'],
                'sesstion_id': response['Data']['SessionId'],
            },
        )

    def _get_stream_request_parameters(self, messages: Messages, parameters: BailianChatParameters) -> HttpxPostKwargs:
        http_post_kwargs = self._get_request_parameters(messages, parameters)
        http_post_kwargs['headers']['Accept'] = 'text/event-stream'  # type: ignore
        http_post_kwargs['json']['Stream'] = True  # type: ignore
        return http_post_kwargs

    @override
    def _stream_completion(self, messages: Messages, parameters: BailianChatParameters) -> Iterator[ChatCompletionStreamOutput]:
        request_parameters = self._get_stream_request_parameters(messages, parameters)
        yield ChatCompletionStreamOutput(
            model_info=self.model_info,
            stream=Stream(delta='', control='start'),
        )
        current_length = 0
        reply = ''
        for line in self.http_client.stream_post(request_parameters=request_parameters):
            output, current_length = self._parse_stream_line(line, current_length)
            reply += output.stream.delta
            if output.is_finish:
                output.messages = [AssistantMessage(content=reply)]
            yield output
            if output.is_finish:
                break

    @override
    async def _async_stream_completion(
        self, messages: Messages, parameters: BailianChatParameters
    ) -> AsyncIterator[ChatCompletionStreamOutput]:
        request_parameters = self._get_stream_request_parameters(messages, parameters)
        yield ChatCompletionStreamOutput(
            model_info=self.model_info,
            stream=Stream(delta='', control='start'),
        )
        current_length = 0
        reply = ''
        async for line in self.http_client.async_stream_post(request_parameters=request_parameters):
            output, current_length = self._parse_stream_line(line, current_length)
            reply += output.stream.delta
            if output.is_finish:
                output.messages = [AssistantMessage(content=reply)]
            yield output
            if output.is_finish:
                break

    def _parse_stream_line(self, line: str, current_length: int) -> tuple[ChatCompletionStreamOutput, int]:
        parsed_line = json.loads(line)
        message: str = parsed_line['Data']['Text']
        if len(message) == current_length:
            output = ChatCompletionStreamOutput(
                model_info=self.model_info,
                extra={
                    'thoughts': parsed_line['Data']['Thoughts'],
                    'doc_references': parsed_line['Data']['DocReferences'],
                    'request_id': parsed_line['RequestId'],
                    'response_id': parsed_line['Data']['ResponseId'],
                },
                stream=Stream(delta='', control='finish'),
            )
            return output, len(message)

        delta = message[current_length:]
        output = ChatCompletionStreamOutput(
            model_info=self.model_info,
            stream=Stream(delta=delta, control='continue'),
        )
        return output, len(message)

    @property
    @override
    def name(self) -> str:
        return self.settings.app_id

    @classmethod
    @override
    def from_name(cls, name: str, **kwargs: Any) -> Self:
        if name:
            raise ValueError(f'{cls} cannot be initialized from name')
        return cls(**kwargs)
