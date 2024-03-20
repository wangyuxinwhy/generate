from __future__ import annotations

import base64
import json
from typing import Any, AsyncIterator, ClassVar, Dict, Iterator, List, Literal, Optional

from pydantic import Field, PositiveInt
from typing_extensions import Annotated, TypedDict, Unpack, override

from generate.chat_completion.base import RemoteChatCompletionModel
from generate.chat_completion.message import Prompt
from generate.chat_completion.message.core import (
    AssistantMessage,
    ImagePart,
    ImageUrlPart,
    Message,
    SystemMessage,
    TextPart,
    UserMessage,
    UserMultiPartMessage,
)
from generate.chat_completion.message.exception import MessageTypeError
from generate.chat_completion.message.utils import ensure_messages
from generate.chat_completion.model_output import ChatCompletionOutput, ChatCompletionStreamOutput
from generate.chat_completion.stream_manager import StreamManager
from generate.http import HttpClient, HttpxPostKwargs
from generate.model import ModelParameters, RemoteModelParametersDict
from generate.platforms import AnthropicSettings
from generate.types import Probability, Temperature


class AnthropicMessage(TypedDict):
    role: Literal['user', 'assistant']
    content: str


class AnthropicChatParameters(ModelParameters):
    system: Optional[str] = None
    max_tokens: PositiveInt = 1024
    metadata: Optional[Dict[str, Any]] = {}
    stop: Annotated[Optional[List[str]], Field(alias='stop_sequences')] = None
    temperature: Optional[Temperature] = None
    top_p: Optional[Probability] = None
    top_k: Optional[PositiveInt] = None


class AnthropicParametersDict(RemoteModelParametersDict, total=False):
    system: Optional[str]
    max_tokens: PositiveInt
    metadata: Optional[Dict[str, Any]]
    stop: Optional[List[str]]
    temperature: Optional[Temperature]
    top_p: Optional[Probability]
    top_k: Optional[PositiveInt]


class AnthropicChat(RemoteChatCompletionModel):
    model_type: ClassVar[str] = 'anthropic'
    available_models: ClassVar[List[str]] = [
        'claude-2.1',
        'claude-2.0',
        'claude-instant-1.2',
        'claude-3-opus-20240229',
        'claude-3-sonnet-20240229',
        'claude-3-haiku-20240307',
    ]

    parameters: AnthropicChatParameters
    settings: AnthropicSettings

    def __init__(
        self,
        model: str = 'claude-2.1',
        parameters: AnthropicChatParameters | None = None,
        settings: AnthropicSettings | None = None,
        http_client: HttpClient | None = None,
    ) -> None:
        parameters = parameters or AnthropicChatParameters()
        settings = settings or AnthropicSettings()  # type: ignore
        http_client = http_client or HttpClient()
        super().__init__(model=model, parameters=parameters, settings=settings, http_client=http_client)

    @override
    def generate(self, prompt: Prompt, **kwargs: Unpack[AnthropicParametersDict]) -> ChatCompletionOutput:
        return super().generate(prompt, **kwargs)

    @override
    async def async_generate(self, prompt: Prompt, **kwargs: Unpack[AnthropicParametersDict]) -> ChatCompletionOutput:
        return await super().async_generate(prompt, **kwargs)

    @override
    def stream_generate(
        self, prompt: Prompt, **kwargs: Unpack[AnthropicParametersDict]
    ) -> Iterator[ChatCompletionStreamOutput]:
        yield from super().stream_generate(prompt, **kwargs)

    @override
    async def async_stream_generate(
        self, prompt: Prompt, **kwargs: Unpack[AnthropicParametersDict]
    ) -> AsyncIterator[ChatCompletionStreamOutput]:
        async for output in super().async_stream_generate(prompt, **kwargs):
            yield output

    def _convert_message(self, message: Message) -> dict[str, str]:
        if isinstance(message, UserMessage):
            return {'role': 'user', 'content': message.content}
        if isinstance(message, AssistantMessage):
            return {'role': 'assistant', 'content': message.content}
        if isinstance(message, UserMultiPartMessage):
            message_dict = {'role': 'user', 'content': []}
            for part in message.content:
                if isinstance(part, TextPart):
                    message_dict['content'].append({'type': 'text', 'text': part.text})

                if isinstance(part, ImagePart):
                    data = base64.b64encode(part.image).decode()
                    media_type = 'image/jpeg' if part.image_format is None else f'image/{part.image_format}'
                    message_dict['content'].append(
                        {'type': 'image', 'source': {'type': 'base64', 'media_type': media_type, 'data': data}}
                    )

                if isinstance(part, ImageUrlPart):
                    response = self.http_client.get({'url': part.image_url.url})
                    data = base64.b64encode(response.content).decode()
                    media_type = response.headers.get('Content-Type') or 'image/jpeg'
                    message_dict['content'].append(
                        {'type': 'image', 'source': {'type': 'base64', 'media_type': media_type, 'data': data}}
                    )
            return message_dict
        raise MessageTypeError(message, (UserMessage, AssistantMessage, UserMultiPartMessage))

    @override
    def _get_request_parameters(
        self, prompt: Prompt, stream: bool = False, **kwargs: Unpack[AnthropicParametersDict]
    ) -> HttpxPostKwargs:
        messages = ensure_messages(prompt)
        parameters = self.parameters.clone_with_changes(**kwargs)

        if isinstance(messages[0], SystemMessage):
            parameters.system = messages[0].content
            messages = messages[1:]

        anthropic_messages = [self._convert_message(message) for message in messages]
        json_dict = parameters.custom_model_dump()
        json_dict['model'] = self.model
        json_dict['messages'] = anthropic_messages
        if stream:
            json_dict['stream'] = True

        headers = {
            'Content-Type': 'application/json',
            'anthropic-version': self.settings.api_version,
            'x-api-key': self.settings.api_key.get_secret_value(),
        }
        return {
            'url': self.settings.api_base + '/messages',
            'headers': headers,
            'json': json_dict,
        }

    @override
    def _process_reponse(self, response: dict[str, Any]) -> ChatCompletionOutput:
        content = ''
        for i in response['content']:
            if i['type'] == 'text':
                content += i['text']
        return ChatCompletionOutput(
            model_info=self.model_info,
            message=AssistantMessage(content=content),
            finish_reason=response['stop_reason'],
            cost=self._calculate_cost(**response['usage']),
            extra={'usage': response['usage'], 'message_id': response['id']},
        )

    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float | None:
        model_price_mapping = {
            'claude-instant': (0.80, 2.40),
            'claude-2': (8, 24),
            'claude-3-haiku': (0.25, 1.25),
            'claude-3-sonnet': (3, 15),
            'claude-3-opus': (15, 75),
        }
        dollar_to_yuan = 7
        for model_name, (prompt_price, completion_price) in model_price_mapping.items():
            if model_name in self.model:
                cost = (input_tokens * prompt_price / 1_000_000) + (output_tokens * completion_price / 1_000_000)
                return cost * dollar_to_yuan
        return None

    @override
    def _process_stream_line(self, line: str, stream_manager: StreamManager) -> ChatCompletionStreamOutput | None:
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            return None

        if 'message' in data:
            input_tokens = data['message']['usage']['input_tokens']
            stream_manager.extra.setdefault('usage', {}).update({'input_tokens': input_tokens})
            return None

        if 'delta' in data:
            if 'stop_reason' in data['delta']:
                delta_dict = data['delta']
                stream_manager.delta = ''
                stream_manager.finish_reason = delta_dict['stop_reason']
                stream_manager.extra['usage']['output_tokens'] = data['usage']['output_tokens']
                stream_manager.cost = self._calculate_cost(**stream_manager.extra['usage'])
                return stream_manager.build_stream_output()

            stream_manager.delta = data['delta']['text']
            return stream_manager.build_stream_output()
        return None
