from __future__ import annotations

import json
from typing import Any, AsyncIterator, ClassVar, Dict, Iterator, List, Optional

from pydantic import Field, PositiveInt
from typing_extensions import Annotated, Self, Unpack, override

from generate.chat_completion.base import ChatCompletionModel
from generate.chat_completion.message import Prompt
from generate.chat_completion.message.core import AssistantMessage, Message, Messages, SystemMessage, UserMessage
from generate.chat_completion.message.exception import MessageTypeError
from generate.chat_completion.message.utils import ensure_messages
from generate.chat_completion.model_output import ChatCompletionOutput, ChatCompletionStreamOutput, Stream
from generate.http import HttpClient, HttpxPostKwargs, ResponseValue
from generate.model import ModelParameters, ModelParametersDict
from generate.platforms import AnthropicSettings
from generate.types import Probability, Temperature


class AnthropicChatParameters(ModelParameters):
    system: Optional[str] = None
    max_tokens: PositiveInt = 1024
    metadata: Optional[Dict[str, Any]] = {}
    stop: Annotated[Optional[List[str]], Field(alias='stop_sequences')] = None
    temperature: Optional[Temperature] = None
    top_p: Optional[Probability] = None
    top_k: Optional[PositiveInt] = None


class AnthropicParametersDict(ModelParametersDict, total=False):
    system: str
    max_tokens: int
    metadata: Dict[str, Any]
    stop: List[str]
    temperature: float
    top_p: float
    top_k: int


class AnthropicChat(ChatCompletionModel):
    model_type: ClassVar[str] = 'anthropic'

    parameters: AnthropicChatParameters
    settings: AnthropicSettings

    def __init__(
        self,
        model: str = 'claude-2.1',
        parameters: AnthropicChatParameters | None = None,
        settings: AnthropicSettings | None = None,
        http_client: HttpClient | None = None,
    ) -> None:
        self.parameters = parameters or AnthropicChatParameters()
        self.settings = settings or AnthropicSettings()  # type: ignore
        self.http_client = http_client or HttpClient()
        self.model = model

    def _convert_message(self, message: Message) -> dict[str, str]:
        if isinstance(message, UserMessage):
            return {'role': 'user', 'content': message.content}
        if isinstance(message, AssistantMessage):
            return {'role': 'assistant', 'content': message.content}
        raise MessageTypeError(message, (UserMessage, AssistantMessage))

    def _get_request_parameters(self, messages: Messages, parameters: AnthropicChatParameters) -> HttpxPostKwargs:
        if isinstance(messages[0], SystemMessage):
            parameters.system = messages[0].content
            messages = messages[1:]

        anthropic_messages = [self._convert_message(message) for message in messages]
        json_dict = parameters.custom_model_dump()
        json_dict['model'] = self.model
        json_dict['messages'] = anthropic_messages
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

    def _get_stream_request_parameters(self, messages: Messages, parameters: AnthropicChatParameters) -> HttpxPostKwargs:
        kwargs = self._get_request_parameters(messages, parameters)
        kwargs['json']['stream'] = True
        return kwargs

    def _parse_reponse(self, response: ResponseValue) -> ChatCompletionOutput:
        content = ''
        for i in response['content']:
            if i['type'] == 'text':
                content += i['text']
        return ChatCompletionOutput(
            model_info=self.model_info,
            message=AssistantMessage(content=content),
            finish_reason=response['stop_reason'],
            cost=self.calculate_cost(**response['usage']),
            extra={'usage': response['usage'], 'message_id': response['id']},
        )

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float | None:
        dollar_to_yuan = 7
        if 'claude-instant' in self.model:
            # prompt: $0.80/million tokens, completion: $2.40/million tokens
            cost = (input_tokens * 0.8 / 1_000_000) + (output_tokens * 2.4 / 1_000_000)
            return cost * dollar_to_yuan
        if 'claude-2' in self.model:
            # prompt: $8/million tokens, completion: $24/million tokens
            cost = (input_tokens * 8 / 1_000_000) + (output_tokens * 24 / 1_000_000)
            return cost * dollar_to_yuan
        return None

    def _parse_stream_line(
        self, line: str, message: AssistantMessage, is_start: bool, usage: dict[str, int]
    ) -> ChatCompletionStreamOutput | None:
        parsed_line = json.loads(line)
        if 'message' in parsed_line:
            message_dict = parsed_line['message']
            usage['input_tokens'] = message_dict['usage']['input_tokens']
            return None
        if 'delta' in parsed_line:
            if 'stop_reason' in parsed_line['delta']:
                delta_dict = parsed_line['delta']
                usage['output_tokens'] = parsed_line['usage']['output_tokens']
                return ChatCompletionStreamOutput(
                    model_info=self.model_info,
                    message=message,
                    finish_reason=delta_dict['stop_reason'],
                    cost=self.calculate_cost(**usage),
                    stream=Stream(delta='', control='finish'),
                    extra={'usage': usage},
                )
            delta = parsed_line['delta']['text']
            message.content += delta
            return ChatCompletionStreamOutput(
                model_info=self.model_info,
                message=message,
                finish_reason=None,
                stream=Stream(delta=delta, control='start' if is_start else 'continue'),
            )
        return None

    @override
    def generate(self, prompt: Prompt, **kwargs: Unpack[AnthropicParametersDict]) -> ChatCompletionOutput:
        messages = ensure_messages(prompt)
        parameters = self.parameters.clone_with_changes(**kwargs)
        request_parameters = self._get_request_parameters(messages, parameters)
        response = self.http_client.post(request_parameters=request_parameters)
        return self._parse_reponse(response.json())

    @override
    async def async_generate(self, prompt: Prompt, **kwargs: Unpack[AnthropicParametersDict]) -> ChatCompletionOutput:
        messages = ensure_messages(prompt)
        parameters = self.parameters.clone_with_changes(**kwargs)
        request_parameters = self._get_request_parameters(messages, parameters)
        response = await self.http_client.async_post(request_parameters=request_parameters)
        return self._parse_reponse(response.json())

    @override
    def stream_generate(
        self, prompt: Prompt, **kwargs: Unpack[AnthropicParametersDict]
    ) -> Iterator[ChatCompletionStreamOutput]:
        messages = ensure_messages(prompt)
        parameters = self.parameters.clone_with_changes(**kwargs)
        request_parameters = self._get_stream_request_parameters(messages, parameters)
        message = AssistantMessage(content='')
        is_start = True
        usage = {}
        for line in self.http_client.stream_post(request_parameters=request_parameters):
            output = self._parse_stream_line(line, message, is_start, usage=usage)
            if output is None:
                continue

            is_start = False
            yield output

    @override
    async def async_stream_generate(
        self, prompt: Prompt, **kwargs: Unpack[AnthropicParametersDict]
    ) -> AsyncIterator[ChatCompletionStreamOutput]:
        messages = ensure_messages(prompt)
        parameters = self.parameters.clone_with_changes(**kwargs)
        request_parameters = self._get_stream_request_parameters(messages, parameters)
        message = AssistantMessage(content='')
        is_start = True
        usage = {}
        async for line in self.http_client.async_stream_post(request_parameters=request_parameters):
            output = self._parse_stream_line(line, message, is_start, usage=usage)
            if output is None:
                continue

            is_start = False
            yield output

    @property
    def name(self) -> str:
        return self.model

    @classmethod
    def from_name(cls, name: str) -> Self:
        return cls(model=name)
