from __future__ import annotations

from typing import Any, AsyncIterator, ClassVar, Iterator, List, Optional

from pydantic import Field
from typing_extensions import Annotated, Unpack, override

from generate.chat_completion.base import RemoteChatCompletionModel
from generate.chat_completion.cost_caculator import CostCalculator, GeneralCostCalculator
from generate.chat_completion.message import (
    AssistantMessage,
    Messages,
    Prompt,
    SystemMessage,
    UserMessage,
)
from generate.chat_completion.message.converter import SimpleMessageConverter
from generate.chat_completion.model_output import ChatCompletionOutput, ChatCompletionStreamOutput, FinishReason, Usage
from generate.chat_completion.stream_manager import StreamManager
from generate.http import (
    HttpClient,
    HttpxPostKwargs,
)
from generate.model import ModelParameters, RemoteModelParametersDict
from generate.platforms.baichuan import BaichuanSettings
from generate.types import ModelPrice, Probability, Temperature

BaichuanModelPrice: ModelPrice = {
    'Baichuan2-Turbo-192k': (16, 16),
    'Baichuan2-Turbo': (8, 8),
}


class BaichuanChatParameters(ModelParameters):
    temperature: Optional[Temperature] = None
    top_k: Optional[Annotated[int, Field(ge=0)]] = None
    top_p: Optional[Probability] = None
    max_tokens: Optional[Annotated[int, Field(ge=0)]] = None
    search: Optional[bool] = Field(default=None, alias='with_search_enhance')


class BaichuanChatParametersDict(RemoteModelParametersDict, total=False):
    temperature: Optional[Temperature]
    top_k: Optional[int]
    top_p: Optional[Probability]
    max_tokens: Optional[int]
    search: Optional[bool]


class BaichuanChat(RemoteChatCompletionModel):
    model_type: ClassVar[str] = 'baichuan'
    available_models: ClassVar[List[str]] = ['Baichuan2-Turbo', 'Baichuan2-Turbo-192k']

    parameters: BaichuanChatParameters
    settings: BaichuanSettings
    message_converter: SimpleMessageConverter

    def __init__(
        self,
        model: str = 'Baichuan2-Turbo',
        parameters: BaichuanChatParameters | None = None,
        settings: BaichuanSettings | None = None,
        http_client: HttpClient | None = None,
        message_converter: SimpleMessageConverter | None = None,
        cost_calculator: CostCalculator | None = None,
    ) -> None:
        parameters = parameters or BaichuanChatParameters()
        settings = settings or BaichuanSettings()  # type: ignore
        http_client = http_client or HttpClient()
        message_converter = message_converter or SimpleMessageConverter()
        cost_calculator = cost_calculator or GeneralCostCalculator(BaichuanModelPrice)
        super().__init__(
            model=model,
            parameters=parameters,
            settings=settings,
            http_client=http_client,
            message_converter=message_converter,
            cost_calculator=cost_calculator,
        )

    @override
    def generate(self, prompt: Prompt, **kwargs: Unpack[BaichuanChatParametersDict]) -> ChatCompletionOutput:
        return super().generate(prompt, **kwargs)

    @override
    async def async_generate(self, prompt: Prompt, **kwargs: Unpack[BaichuanChatParametersDict]) -> ChatCompletionOutput:
        return await super().async_generate(prompt, **kwargs)

    @override
    def stream_generate(
        self, prompt: Prompt, **kwargs: Unpack[BaichuanChatParametersDict]
    ) -> Iterator[ChatCompletionStreamOutput]:
        yield from super().stream_generate(prompt, **kwargs)

    @override
    async def async_stream_generate(
        self, prompt: Prompt, **kwargs: Unpack[BaichuanChatParametersDict]
    ) -> AsyncIterator[ChatCompletionStreamOutput]:
        async for output in super().async_stream_generate(prompt, **kwargs):
            yield output

    @override
    def _get_request_parameters(
        self, messages: Messages, stream: bool = False, **kwargs: Unpack[BaichuanChatParametersDict]
    ) -> HttpxPostKwargs:
        if isinstance(system_message := messages[0], SystemMessage):
            prepend_messages = [UserMessage(content=system_message.content)]
            messages = prepend_messages + messages[1:]
        parameters = self.parameters.clone_with_changes(**kwargs)
        json_data = {
            'model': self.model,
            'messages': self.message_converter.convert_messages(messages),
        }
        parameters_dict = parameters.custom_model_dump()
        json_data.update(parameters_dict)
        if stream:
            json_data['stream'] = True
        headers = {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer ' + self.settings.api_key.get_secret_value(),
        }
        return {
            'url': self.settings.api_base + '/chat/completions',
            'headers': headers,
            'json': json_data,
        }

    @override
    def _process_reponse(self, response: dict[str, Any]) -> ChatCompletionOutput:
        return ChatCompletionOutput(
            model_info=self.model_info,
            message=self._parse_assistant_message(response),
            finish_reason=self._parse_finish_reason(response),
            usage=self._parse_usage(response),
            extra=self._parse_extra(response),
        )

    @override
    def _process_stream_response(
        self, response: dict[str, Any], stream_manager: StreamManager
    ) -> ChatCompletionStreamOutput | None:
        stream_manager.delta = response['choices'][0]['delta']['content']
        stream_manager.finish_reason = self._parse_finish_reason(response)
        stream_manager.extra = self._parse_extra(response)
        stream_manager.usage = self._parse_usage(response)
        return stream_manager.build_stream_output()

    def _parse_assistant_message(self, response: dict[str, Any]) -> AssistantMessage:
        return AssistantMessage(content=response['choices'][0]['message']['content'])

    def _parse_usage(self, response: dict[str, Any]) -> Usage:
        usage = response.get('usage')
        if usage is not None:
            input_tokens = usage['prompt_tokens']
            output_tokens = usage['completion_tokens']
            cost = self.cost(input_tokens, output_tokens)
            return Usage(input_tokens=input_tokens, output_tokens=output_tokens, cost=cost)
        return Usage()

    def _parse_finish_reason(self, response: dict[str, Any]) -> FinishReason | None:
        try:
            choice = response['choices'][0]
            if finish_reason := choice.get('finish_reason'):
                return FinishReason(finish_reason)
        except (KeyError, IndexError, ValueError):
            return None

    def _parse_extra(self, response: dict[str, Any]) -> dict[str, Any]:
        return {'response': response}
