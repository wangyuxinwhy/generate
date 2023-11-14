from __future__ import annotations

import os
import uuid
from contextvars import ContextVar
from typing import Any, ClassVar, List, Literal, Optional

from pydantic import Field
from typing_extensions import NotRequired, Self, TypedDict, Unpack, override

from generate.chat_completion.http_chat import (
    HttpChatModel,
    HttpModelInitKwargs,
    HttpResponse,
    HttpxPostKwargs,
    UnexpectedResponseError,
)
from generate.chat_completion.message import (
    AssistantMessage,
    Messages,
    MessageTypeError,
    UserMessage,
)
from generate.chat_completion.model_output import ChatCompletionModelOutput, FinishStream, Stream
from generate.parameters import ModelParameters
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
    pairs = []
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
    parameters: Optional[BailianParameter] = Field(default=None, alias='Parameters')
    doc_tag_ids: Optional[List[int]] = Field(default=None, alias='DocTagIds')


class BailianChat(HttpChatModel[BailianChatParameters]):
    model_type: ClassVar[str] = 'bailian'
    default_api: ClassVar[str] = 'https://bailian.aliyuncs.com/v2/app/completions'

    def __init__(
        self,
        app_id: str | None = None,
        access_key_id: str | None = None,
        access_key_secret: str | None = None,
        agent_key: str | None = None,
        api: str | None = None,
        parameters: BailianChatParameters | None = None,
        **kwargs: Unpack[HttpModelInitKwargs],
    ) -> None:
        try:
            import broadscope_bailian
        except ImportError as e:
            raise ImportError('Please install broadscope_bailian first: pip install broadscope_bailian') from e

        parameters = parameters or BailianChatParameters()
        super().__init__(parameters=parameters, **kwargs)
        self.app_id = app_id or os.environ['BAILIAN_APP_ID']
        self.access_key_id = access_key_id or os.environ['BAILIAN_ACCESS_KEY_ID']
        self.access_key_secret = access_key_secret or os.environ['BAILIAN_ACCESS_KEY_SECRET']
        self.agent_key = agent_key or os.environ['BAILIAN_AGENT_KEY']
        self.api = api or self.default_api
        client = broadscope_bailian.AccessTokenClient(
            access_key_id=self.access_key_id,
            access_key_secret=self.access_key_secret,
            agent_key=self.agent_key,
        )
        self.token = client.get_token()
        self._stream_length = ContextVar('stream_length', default=0)

    @override
    def _get_request_parameters(self, messages: Messages, parameters: BailianChatParameters) -> HttpxPostKwargs:
        if not isinstance(messages[-1], UserMessage):
            raise MessageTypeError(messages[-1], allowed_message_type=(UserMessage,))

        prompt = messages[-1].content
        history = convert_to_bailian_chat_qa_pair(messages[:-1])

        json_dict = parameters.model_dump(exclude_none=True, by_alias=True)
        headers = {
            'Content-Type': 'application/json;charset=UTF-8',
            'Authorization': f'Bearer {self.token}',
        }
        json_dict['Prompt'] = prompt
        json_dict['AppId'] = self.app_id
        json_dict['History'] = history
        return {
            'url': self.api,
            'headers': headers,
            'json': json_dict,
        }

    @override
    def _get_stream_request_parameters(self, messages: Messages, parameters: BailianChatParameters) -> HttpxPostKwargs:
        http_post_kwargs = self._get_request_parameters(messages, parameters)
        http_post_kwargs['headers']['Accept'] = 'text/event-stream'  # type: ignore
        http_post_kwargs['json']['Stream'] = True  # type: ignore
        return http_post_kwargs

    @override
    def _parse_stream_response(self, response: HttpResponse) -> Stream:
        message: str = response['Data']['Text']
        if len(message) == self._stream_length.get():
            return FinishStream(
                extra={
                    'thoughts': response['Data']['Thoughts'],
                    'doc_references': response['Data']['DocReferences'],
                    'request_id': response['RequestId'],
                    'response_id': response['Data']['ResponseId'],
                }
            )

        delta = message[self._stream_length.get() :]
        self._stream_length.set(len(message))
        return Stream(delta=delta, control='continue')

    @override
    def _parse_reponse(self, response: HttpResponse) -> ChatCompletionModelOutput:
        if not response['Success']:
            raise UnexpectedResponseError(response)
        messages = [AssistantMessage(content=response['Data']['Text'])]
        return ChatCompletionModelOutput(
            chat_model_id=self.model_id,
            messages=messages,
            extra={
                'thoughts': response['Data']['Thoughts'],
                'doc_references': response['Data']['DocReferences'],
                'request_id': response['RequestId'],
                'response_id': response['Data']['ResponseId'],
                'sesstion_id': response['Data']['SessionId'],
            },
        )

    @property
    @override
    def name(self) -> str:
        return self.app_id

    @classmethod
    @override
    def from_name(cls, name: str, **kwargs: Any) -> Self:
        return cls(app_id=name, **kwargs)
