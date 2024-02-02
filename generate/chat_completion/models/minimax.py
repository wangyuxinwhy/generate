from __future__ import annotations

import json
from typing import Any, AsyncIterator, ClassVar, Iterator, Literal, Optional

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
from generate.platforms.minimax import MinimaxSettings
from generate.types import Probability, Temperature


class MinimaxMessage(TypedDict):
    sender_type: Literal['USER', 'BOT']
    text: str


class RoleMeta(TypedDict):
    user_name: str
    bot_name: str


DEFAULT_MINIMAX_SYSTEM_PROMPT = 'MM智能助理是一款由MiniMax自研的，没有调用其他产品的接口的大型语言模型。MiniMax是一家中国科技公司，一直致力于进行大模型相关的研究。'


class MinimaxChatParameters(ModelParameters):
    system_prompt: str = Field(default=DEFAULT_MINIMAX_SYSTEM_PROMPT, serialization_alias='prompt')
    role_meta: RoleMeta = {'user_name': '用户', 'bot_name': 'MM智能助理'}
    beam_width: Optional[Annotated[int, Field(ge=1, le=4)]] = None
    temperature: Optional[Temperature] = None
    top_p: Optional[Probability] = None
    max_tokens: Optional[Annotated[PositiveInt, Field(serialization_alias='tokens_to_generate')]] = None
    skip_info_mask: Optional[bool] = None
    continue_last_message: Optional[bool] = None

    def custom_model_dump(self) -> dict[str, Any]:
        output = super().custom_model_dump()
        if 'temperature' in output:
            output['temperature'] = max(0.01, output['temperature'])
        if 'top_p' in output:
            output['top_p'] = max(0.01, output['top_p'])
        return output


class MinimaxChatParametersDict(ModelParametersDict, total=False):
    system_prompt: str
    role_meta: RoleMeta
    beam_width: Optional[int]
    temperature: Optional[Temperature]
    top_p: Optional[Probability]
    max_tokens: Optional[int]
    skip_info_mask: Optional[bool]
    continue_last_message: Optional[bool]


def _convert_message_to_minimax_message(message: Message) -> MinimaxMessage:
    if isinstance(message, UserMessage):
        return {
            'sender_type': 'USER',
            'text': message.content,
        }
    if isinstance(message, AssistantMessage):
        return {
            'sender_type': 'BOT',
            'text': message.content,
        }
    raise MessageTypeError(message, (UserMessage, AssistantMessage))


def _convert_messages(messages: Messages) -> list[MinimaxMessage]:
    if isinstance(system_message := messages[0], SystemMessage):
        prepend_messages = [UserMessage(content=system_message.content), AssistantMessage(content='好的')]
        messages = prepend_messages + messages[1:]
    return [_convert_message_to_minimax_message(message) for message in messages]


class MinimaxChat(RemoteChatCompletionModel):
    model_type: ClassVar[str] = 'minimax'

    parameters: MinimaxChatParameters
    settings: MinimaxSettings

    def __init__(
        self,
        model: str = 'abab5.5-chat',
        settings: MinimaxSettings | None = None,
        parameters: MinimaxChatParameters | None = None,
        http_client: HttpClient | None = None,
    ) -> None:
        parameters = parameters or MinimaxChatParameters()
        settings = settings or MinimaxSettings()  # type: ignore
        http_client = http_client or HttpClient()
        super().__init__(parameters=parameters, settings=settings, http_client=http_client)

        self.model = model

    def _get_request_parameters(self, messages: Messages, parameters: MinimaxChatParameters) -> HttpxPostKwargs:
        minimax_messages = _convert_messages(messages)
        parameters_dict = parameters.custom_model_dump()
        json_data = {
            'model': self.model,
            'messages': minimax_messages,
            **parameters_dict,
        }
        headers = {
            'Authorization': f'Bearer {self.settings.api_key.get_secret_value()}',
            'Content-Type': 'application/json',
        }
        return {
            'url': self.settings.api_base + 'text/chatcompletion',
            'json': json_data,
            'headers': headers,
            'params': {'GroupId': self.settings.group_id},
        }

    @override
    def generate(self, prompt: Prompt, **kwargs: Unpack[MinimaxChatParametersDict]) -> ChatCompletionOutput:
        messages = ensure_messages(prompt)
        parameters = self.parameters.clone_with_changes(**kwargs)
        request_parameters = self._get_request_parameters(messages, parameters)
        response = self.http_client.post(request_parameters=request_parameters)
        return self._parse_reponse(response.json())

    @override
    async def async_generate(self, prompt: Prompt, **kwargs: Unpack[MinimaxChatParametersDict]) -> ChatCompletionOutput:
        messages = ensure_messages(prompt)
        parameters = self.parameters.clone_with_changes(**kwargs)
        request_parameters = self._get_request_parameters(messages, parameters)
        response = await self.http_client.async_post(request_parameters=request_parameters)
        return self._parse_reponse(response.json())

    def _parse_reponse(self, response: ResponseValue) -> ChatCompletionOutput:
        try:
            return ChatCompletionOutput(
                model_info=self.model_info,
                message=AssistantMessage(content=response['choices'][0]['text']),
                finish_reason=response['choices'][0]['finish_reason'],
                cost=self.calculate_cost(response['usage']),
                extra={
                    'logprobes': response['choices'][0]['logprobes'],
                    'input_sensitive': False,
                    'output_sensitive': False,
                    'usage': response['usage'],
                },
            )
        except (KeyError, IndexError, TypeError) as e:
            raise UnexpectedResponseError(response) from e

    def _get_stream_request_parameters(self, messages: Messages, parameters: MinimaxChatParameters) -> HttpxPostKwargs:
        http_parameters = self._get_request_parameters(messages, parameters)
        http_parameters['json']['stream'] = True
        http_parameters['json']['use_standard_sse'] = True
        return http_parameters

    @override
    def stream_generate(
        self, prompt: Prompt, **kwargs: Unpack[MinimaxChatParametersDict]
    ) -> Iterator[ChatCompletionStreamOutput]:
        messages = ensure_messages(prompt)
        parameters = self.parameters.clone_with_changes(**kwargs)
        request_parameters = self._get_stream_request_parameters(messages, parameters)
        message = AssistantMessage(content='')
        is_start = True
        for line in self.http_client.stream_post(request_parameters=request_parameters):
            output = self._parse_stream_line(line, message, is_start)
            is_start = False
            yield output

    @override
    async def async_stream_generate(
        self, prompt: Prompt, **kwargs: Unpack[MinimaxChatParametersDict]
    ) -> AsyncIterator[ChatCompletionStreamOutput]:
        messages = ensure_messages(prompt)
        parameters = self.parameters.clone_with_changes(**kwargs)
        request_parameters = self._get_stream_request_parameters(messages, parameters)
        message = AssistantMessage(content='')
        is_start = True
        async for line in self.http_client.async_stream_post(request_parameters=request_parameters):
            output = self._parse_stream_line(line, message, is_start)
            is_start = False
            yield output

    def _parse_stream_line(self, line: str, message: AssistantMessage, is_start: bool) -> ChatCompletionStreamOutput:
        parsed_line = json.loads(line)
        delta = parsed_line['choices'][0]['delta']
        message.content += delta
        if parsed_line['reply']:
            return ChatCompletionStreamOutput(
                model_info=self.model_info,
                finish_reason=parsed_line['choices'][0]['finish_reason'],
                message=message,
                cost=self.calculate_cost(parsed_line['usage']),
                extra={
                    'logprobes': parsed_line['choices'][0]['logprobes'],
                    'input_sensitive': False,
                    'output_sensitive': False,
                    'usage': parsed_line['usage'],
                },
                stream=Stream(delta=delta, control='finish'),
            )
        return ChatCompletionStreamOutput(
            model_info=self.model_info,
            finish_reason=None,
            message=message,
            stream=Stream(delta=delta, control='start' if is_start else 'continue'),
        )

    def calculate_cost(self, usage: dict[str, int]) -> float:
        return 0.015 * (usage['total_tokens'] / 1000)

    @property
    @override
    def name(self) -> str:
        return self.model

    @classmethod
    @override
    def from_name(cls, name: str) -> Self:
        return cls(model=name)
