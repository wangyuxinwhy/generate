from __future__ import annotations

import json
from typing import Any, AsyncIterator, ClassVar, Dict, Iterator, List, Literal, Optional, cast

from pydantic import Field, PositiveInt, model_validator
from typing_extensions import Annotated, NotRequired, Self, TypedDict, Unpack, override

from generate.chat_completion.base import RemoteChatCompletionModel
from generate.chat_completion.function_call import FunctionJsonSchema
from generate.chat_completion.message import (
    AssistantMessage,
    FunctionCall,
    FunctionMessage,
    Message,
    Messages,
    MessageTypeError,
    MessageValueError,
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
from generate.model import ModelInfo, ModelParameters, ModelParametersDict
from generate.platforms.minimax import MinimaxSettings
from generate.types import Probability, Temperature


class BotSettingDict(TypedDict):
    bot_name: str
    content: str


class GlyphDict(TypedDict):
    type: str
    raw_glpyh: str
    json_properties: Dict[str, Any]


class ReplyConstrainsDict(TypedDict):
    sender_type: str
    sender_name: str
    glyph: NotRequired[GlyphDict]


class MinimaxFunctionCall(TypedDict):
    name: str
    arguments: str


class MinimaxProMessage(TypedDict):
    sender_type: Literal['USER', 'BOT', 'FUNCTION']
    sender_name: str
    text: str
    function_call: NotRequired[MinimaxFunctionCall]


class MinimaxProChatParameters(ModelParameters):
    reply_constraints: ReplyConstrainsDict = {'sender_type': 'BOT', 'sender_name': 'MM智能助理'}
    bot_setting: List[BotSettingDict] = [
        {
            'bot_name': 'MM智能助理',
            'content': 'MM智能助理是一款由MiniMax自研的，没有调用其他产品的接口的大型语言模型。MiniMax是一家中国科技公司，一直致力于进行大模型相关的研究。',
        }
    ]
    temperature: Optional[Temperature] = None
    top_p: Optional[Probability] = None
    max_tokens: Annotated[Optional[PositiveInt], Field(serialization_alias='tokens_to_generate')] = None
    mask_sensitive_info: Optional[bool] = None
    sample_messages: Optional[List[MinimaxProMessage]] = None
    functions: Optional[List[FunctionJsonSchema]] = None
    search: Optional[bool] = None
    plugins: Optional[List[str]] = None

    @model_validator(mode='after')
    def check_bot_name(self) -> Self:
        names: set[str] = {bot_setting['bot_name'] for bot_setting in self.bot_setting}
        if (sender_name := self.reply_constraints['sender_name']) not in names:
            raise ValueError(f'reply_constraints sender_name {sender_name} must be in bot_setting names: {names}')
        return self

    @property
    def bot_name(self) -> str | None:
        if len(self.bot_setting) == 1:
            return self.bot_setting[0]['bot_name']
        return None

    def set_system_prompt(self, system_prompt: str) -> None:
        if len(self.bot_setting) == 1:
            self.bot_setting[0]['content'] = system_prompt
        else:
            raise ValueError('set system_prompt is not supported when bot_setting has more than one bot')

    def set_bot_name(self, bot_name: str) -> None:
        if len(self.bot_setting) == 1:
            self.bot_setting[0]['bot_name'] = bot_name
            self.reply_constraints['sender_name'] = bot_name
        else:
            raise ValueError('set bot_name is not supported when bot_setting has more than one bot')

    def custom_model_dump(self) -> dict[str, Any]:
        output = super().custom_model_dump()
        if 'temperature' in output:
            output['temperature'] = max(0.01, output['temperature'])
        if 'top_p' in output:
            output['top_p'] = max(0.01, output['top_p'])
        if 'search' in output:
            original_plugins = output.get('plugins', [])
            output['plugins'] = list(set(original_plugins + ['plugin_web_search']))
        return output


class MinimaxProChatParametersDict(ModelParametersDict, total=False):
    reply_constraints: ReplyConstrainsDict
    bot_setting: List[BotSettingDict]
    temperature: Optional[Temperature]
    top_p: Optional[Probability]
    max_tokens: Optional[PositiveInt]
    mask_sensitive_info: Optional[bool]
    sample_messages: Optional[List[MinimaxProMessage]]
    functions: Optional[List[FunctionJsonSchema]]
    search: Optional[bool]
    plugins: Optional[List[str]]


def _convert_to_minimax_pro_message(
    message: Message, default_bot_name: str | None = None, default_user_name: str = '用户'
) -> MinimaxProMessage:
    if isinstance(message, UserMessage):
        sender_name = message.name or default_user_name
        return {'sender_type': 'USER', 'sender_name': sender_name, 'text': message.content}

    if isinstance(message, AssistantMessage):
        sender_name = message.name or default_bot_name
        if sender_name is None:
            raise MessageValueError(message, 'bot name is required')
        if message.function_call is None:
            return {
                'sender_type': 'BOT',
                'sender_name': sender_name,
                'text': message.content,
            }
        return {
            'sender_type': 'BOT',
            'sender_name': sender_name,
            'text': message.content,
            'function_call': {
                'name': message.function_call.name,
                'arguments': message.function_call.arguments,
            },
        }

    if isinstance(message, FunctionMessage):
        return {
            'sender_type': 'FUNCTION',
            'sender_name': message.name,
            'text': message.content,
        }

    raise MessageTypeError(message, allowed_message_type=(UserMessage, AssistantMessage, FunctionMessage))


def _convert_to_message(message: MinimaxProMessage) -> AssistantMessage | FunctionMessage:
    if 'function_call' in message:
        return AssistantMessage(
            name=message['sender_name'],
            content=message['text'],
            function_call=FunctionCall(name=message['function_call']['name'], arguments=message['function_call']['arguments']),
        )
    if message['sender_type'] == 'BOT':
        return AssistantMessage(
            name=message['sender_name'],
            content=message['text'],
        )
    if message['sender_type'] == 'FUNCTION':
        return FunctionMessage(
            name=message['sender_name'],
            content=message['text'],
        )
    raise ValueError(f'unknown sender_type: {message["sender_type"]}')


class _StreamResponseProcessor:
    def __init__(self, model_info: ModelInfo) -> None:
        self.message: AssistantMessage | None = None
        self.model_info = model_info

    def process(self, response: ResponseValue) -> ChatCompletionStreamOutput:
        if response.get('usage'):
            assert self.message is not None
            return ChatCompletionStreamOutput(
                model_info=self.model_info,
                message=self.message,
                finish_reason=response['choices'][0]['finish_reason'],
                cost=calculate_cost(model_name=self.model_info.name, usage=response['usage']),
                extra={
                    'input_sensitive': response['input_sensitive'],
                    'output_sensitive': response['output_sensitive'],
                    'usage': response['usage'],
                },
                stream=Stream(delta='', control='finish'),
            )

        if self.message is None:
            self.message = self.initial_message(response)
            delta = self.message.content if isinstance(self.message, AssistantMessage) else ''
            control = 'start'
        else:
            delta = self.update_existing_message(response)
            control = 'continue'

        return ChatCompletionStreamOutput(
            model_info=self.model_info,
            message=self.message,
            finish_reason=None,
            stream=Stream(delta=delta, control=control),
        )

    def initial_message(self, response: ResponseValue) -> AssistantMessage:
        output_messages = [_convert_to_message(i) for i in response['choices'][0]['messages']]
        message = output_messages[-1]
        return cast(AssistantMessage, message)

    def update_existing_message(self, response: ResponseValue) -> str:
        output_messages = [_convert_to_message(i) for i in response['choices'][0]['messages']]
        message = output_messages[-1]
        if not isinstance(message, AssistantMessage):
            return ''

        if message.function_call is not None:
            delta = ''
            self.message = message
            return delta

        delta = message.content
        self.message.content += message.content  # type: ignore
        return delta


def calculate_cost(model_name: str, usage: dict[str, int], num_web_search: int = 0) -> float:
    if model_name == 'abab6-chat':
        model_cost = 0.1 * (usage['total_tokens'] / 1000)
    elif model_name == 'abab5.5-chat':
        model_cost = 0.015 * (usage['total_tokens'] / 1000)
    elif model_name == 'abab5.5s-chat':
        model_cost = 0.005 * (usage['total_tokens'] / 1000)
    else:
        return 0.0
    return model_cost + (0.03 * num_web_search)


class MinimaxProChat(RemoteChatCompletionModel):
    model_type: ClassVar[str] = 'minimax_pro'

    parameters: MinimaxProChatParameters
    settings: MinimaxSettings

    def __init__(
        self,
        model: str = 'abab5.5-chat',
        parameters: MinimaxProChatParameters | None = None,
        settings: MinimaxSettings | None = None,
        http_client: HttpClient | None = None,
    ) -> None:
        parameters = parameters or MinimaxProChatParameters()
        settings = settings or MinimaxSettings()  # type: ignore
        http_client = http_client or HttpClient()
        super().__init__(parameters=parameters, settings=settings, http_client=http_client)

        self.model = model
        self.default_user_name = '用户'

    def _get_request_parameters(self, messages: Messages, parameters: MinimaxProChatParameters) -> HttpxPostKwargs:
        if isinstance(messages[0], SystemMessage):
            system_message = messages[0]
            parameters.set_system_prompt(system_message.content)
            messages = messages[1:]

        minimax_pro_messages = [
            _convert_to_minimax_pro_message(
                message, default_bot_name=parameters.bot_name, default_user_name=self.default_user_name
            )
            for message in messages
        ]
        json_data = {'model': self.model, 'messages': minimax_pro_messages, **parameters.custom_model_dump()}
        headers = {
            'Authorization': f'Bearer {self.settings.api_key.get_secret_value()}',
            'Content-Type': 'application/json',
        }
        return {
            'url': self.settings.api_base + 'text/chatcompletion_pro',
            'json': json_data,
            'headers': headers,
            'params': {'GroupId': self.settings.group_id},
        }

    @override
    def generate(self, prompt: Prompt, **kwargs: Unpack[MinimaxProChatParametersDict]) -> ChatCompletionOutput:
        messages = ensure_messages(prompt)
        parameters = self.parameters.clone_with_changes(**kwargs)
        request_parameters = self._get_request_parameters(messages, parameters)
        response = self.http_client.post(request_parameters=request_parameters)
        return self._parse_reponse(response.json())

    @override
    async def async_generate(self, prompt: Prompt, **kwargs: Unpack[MinimaxProChatParametersDict]) -> ChatCompletionOutput:
        messages = ensure_messages(prompt)
        parameters = self.parameters.clone_with_changes(**kwargs)
        request_parameters = self._get_request_parameters(messages, parameters)
        response = await self.http_client.async_post(request_parameters=request_parameters)
        return self._parse_reponse(response.json())

    def _parse_reponse(self, response: ResponseValue) -> ChatCompletionOutput:
        try:
            messages: list[AssistantMessage | FunctionMessage] = [
                _convert_to_message(i) for i in response['choices'][0]['messages']
            ]
            message = cast(AssistantMessage, messages[-1])
            finish_reason = response['choices'][0]['finish_reason']
            num_web_search = sum([1 for i in response['choices'][0]['messages'] if i['sender_name'] == 'plugin_web_search'])
            return ChatCompletionOutput(
                model_info=self.model_info,
                message=message,
                finish_reason=finish_reason,
                cost=calculate_cost(model_name=self.name, usage=response['usage'], num_web_search=num_web_search),
                extra={
                    'input_sensitive': response['input_sensitive'],
                    'output_sensitive': response['output_sensitive'],
                    'usage': response['usage'],
                },
            )
        except (KeyError, IndexError, TypeError) as e:
            raise UnexpectedResponseError(response) from e

    def _get_stream_request_parameters(self, messages: Messages, parameters: MinimaxProChatParameters) -> HttpxPostKwargs:
        http_parameters = self._get_request_parameters(messages, parameters)
        http_parameters['json']['stream'] = True
        return http_parameters

    @override
    def stream_generate(
        self, prompt: Prompt, **kwargs: Unpack[MinimaxProChatParametersDict]
    ) -> Iterator[ChatCompletionStreamOutput]:
        messages = ensure_messages(prompt)
        parameters = self.parameters.clone_with_changes(**kwargs)
        request_parameters = self._get_stream_request_parameters(messages, parameters)
        stream_processor = _StreamResponseProcessor(model_info=self.model_info)
        for line in self.http_client.stream_post(request_parameters=request_parameters):
            yield stream_processor.process(json.loads(line))

    @override
    async def async_stream_generate(
        self, prompt: Prompt, **kwargs: Unpack[MinimaxProChatParametersDict]
    ) -> AsyncIterator[ChatCompletionStreamOutput]:
        messages = ensure_messages(prompt)
        parameters = self.parameters.clone_with_changes(**kwargs)
        request_parameters = self._get_stream_request_parameters(messages, parameters)
        stream_processor = _StreamResponseProcessor(model_info=self.model_info)
        async for line in self.http_client.async_stream_post(request_parameters=request_parameters):
            yield stream_processor.process(json.loads(line))

    @property
    @override
    def name(self) -> str:
        return self.model

    @classmethod
    @override
    def from_name(cls, name: str) -> Self:
        return cls(model=name)
