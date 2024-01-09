from __future__ import annotations

import json
import uuid
from typing import Any, AsyncIterator, ClassVar, Iterator, List, Literal, Optional

from pydantic import Field
from typing_extensions import Annotated, NotRequired, Self, TypedDict, Unpack, override

from generate.chat_completion.base import ChatCompletionModel
from generate.chat_completion.message import (
    AssistantMessage,
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


def _convert_to_bailian_chat_qa_pair(messages: Messages) -> list[BailianChatQAPair]:
    pairs: list[BailianChatQAPair] = []
    if messages and isinstance(messages[0], SystemMessage):
        pairs.append({'User': messages[0].content, 'Bot': '好的'})
        messages = messages[1:]

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


class BailianChatParametersDict(ModelParametersDict, total=False):
    request_id: str
    session_id: Optional[str]
    top_p: Optional[Probability]
    has_thoughts: Optional[bool]
    doc_reference_type: Optional[Literal['indexed', 'simpole']]
    top_k: Optional[int]
    seed: Optional[int]
    use_raw_prompt: Optional[bool]
    doc_tag_ids: Optional[List[int]]


class BailianChat(ChatCompletionModel):
    model_type: ClassVar[str] = 'bailian'

    def __init__(
        self,
        app_id: str | None = None,
        parameters: BailianChatParameters | None = None,
        settings: BailianSettings | None = None,
        http_client: HttpClient | None = None,
    ) -> None:
        self.parameters = parameters or BailianChatParameters()
        self.settings = settings or BailianSettings()  # type: ignore
        self.app_id = app_id or self.settings.default_app_id
        self.http_client = http_client or HttpClient()
        self.token_manager = BailianTokenManager(self.settings, self.http_client)

    def _get_request_parameters(self, messages: Messages, parameters: BailianChatParameters) -> HttpxPostKwargs:
        if not isinstance(messages[-1], UserMessage):
            raise MessageTypeError(messages[-1], allowed_message_type=(UserMessage,))

        prompt = messages[-1].content
        history = _convert_to_bailian_chat_qa_pair(messages[:-1])

        json_dict = parameters.custom_model_dump()
        headers = {
            'Content-Type': 'application/json;charset=UTF-8',
            'Authorization': f'Bearer {self.token_manager.token}',
        }
        json_dict['Prompt'] = prompt
        json_dict['AppId'] = self.app_id
        json_dict['History'] = history
        return {
            'url': self.settings.completion_api,
            'headers': headers,
            'json': json_dict,
        }

    @override
    def generate(self, prompt: Prompt, **kwargs: Unpack[BailianChatParametersDict]) -> ChatCompletionOutput[AssistantMessage]:
        messages = ensure_messages(prompt)
        parameters = self.parameters.update_with_validate(**kwargs)
        request_parameters = self._get_request_parameters(messages, parameters)
        response = self.http_client.post(request_parameters=request_parameters)
        return self._parse_reponse(response.json())

    @override
    async def async_generate(
        self, prompt: Prompt, **kwargs: Unpack[BailianChatParametersDict]
    ) -> ChatCompletionOutput[AssistantMessage]:
        messages = ensure_messages(prompt)
        parameters = self.parameters.update_with_validate(**kwargs)
        request_parameters = self._get_request_parameters(messages, parameters)
        response = await self.http_client.async_post(request_parameters=request_parameters)
        return self._parse_reponse(response.json())

    def _parse_reponse(self, response: ResponseValue) -> ChatCompletionOutput[AssistantMessage]:
        if not response['Success']:
            raise UnexpectedResponseError(response)
        return ChatCompletionOutput(
            model_info=self.model_info,
            message=AssistantMessage(content=response['Data']['Text']),
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
    def stream_generate(
        self, prompt: Prompt, **kwargs: Unpack[BailianChatParametersDict]
    ) -> Iterator[ChatCompletionStreamOutput[AssistantMessage]]:
        messages = ensure_messages(prompt)
        parameters = self.parameters.update_with_validate(**kwargs)
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
    ) -> AsyncIterator[ChatCompletionStreamOutput[AssistantMessage]]:
        messages = ensure_messages(prompt)
        parameters = self.parameters.update_with_validate(**kwargs)
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

    def _parse_stream_line(
        self, line: str, message: AssistantMessage, is_start: bool
    ) -> ChatCompletionStreamOutput[AssistantMessage]:
        parsed_line = json.loads(line)
        reply: str = parsed_line['Data']['Text']
        extra = {
            'thoughts': parsed_line['Data']['Thoughts'],
            'doc_references': parsed_line['Data']['DocReferences'],
            'request_id': parsed_line['RequestId'],
            'response_id': parsed_line['Data']['ResponseId'],
        }
        if len(reply) == len(message.content):
            return ChatCompletionStreamOutput(
                model_info=self.model_info,
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
