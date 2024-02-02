from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import Field, model_validator
from typing_extensions import Annotated, Self, TypedDict, Unpack, override

from generate.http import HttpClient, HttpxPostKwargs, UnexpectedResponseError
from generate.model import ModelParameters
from generate.platforms.minimax import MinimaxSettings
from generate.text_to_speech.base import RemoteTextToSpeechModel, TextToSpeechOutput


class TimeberWeight(TypedDict):
    voice_id: str
    weight: int


class MinimaxSpeechParameters(ModelParameters):
    voice: Optional[str] = Field(default=None, alias='voice_id')
    speed: Annotated[Optional[float], Field(ge=0.5, le=2.0)] = None
    vol: Annotated[Optional[float], Field(gt=0, le=10)] = None
    pitch: Annotated[Optional[float], Field(ge=-12, le=12)] = None
    timber_weights: Annotated[Optional[List[TimeberWeight]], Field(min_length=1, max_length=4)] = None

    @model_validator(mode='after')
    def voice_exists(self) -> Self:
        if self.voice is None and self.timber_weights is None:
            self.voice = 'male-qn-qingse'
        return self


class MinimaxSpeechParametersDict(TypedDict, total=False):
    voice: Optional[str]
    speed: Optional[float]
    vol: Optional[float]
    pitch: Optional[float]
    timber_weights: Optional[List[TimeberWeight]]


class MinimaxProSpeechParameters(MinimaxSpeechParameters):
    audio_sample_rate: Annotated[Optional[int], Field(ge=16000, le=24000)] = 24000
    bitrate: Literal[32000, 64000, 128000] = 128000


class MinimaxProSpeechParametersDict(MinimaxSpeechParametersDict, total=False):
    audio_sample_rate: Optional[int]
    bitrate: Optional[Literal[32000, 64000, 128000]]


class MinimaxSpeech(RemoteTextToSpeechModel):
    model_type = 'minimax'

    parameters: MinimaxSpeechParameters
    settings: MinimaxSettings

    def __init__(
        self,
        model: str = 'speech-01',
        settings: MinimaxSettings | None = None,
        parameters: MinimaxSpeechParameters | None = None,
        http_client: HttpClient | None = None,
    ) -> None:
        parameters = parameters or MinimaxSpeechParameters()
        settings = settings or MinimaxSettings()  # type: ignore
        http_client = http_client or HttpClient()
        super().__init__(parameters=parameters, settings=settings, http_client=http_client)

        self.model = model

    def _get_request_parameters(self, text: str, parameters: MinimaxSpeechParameters) -> HttpxPostKwargs:
        json_data = {
            'model': self.model,
            'text': text,
            **parameters.custom_model_dump(),
        }
        headers = {
            'Authorization': f'Bearer {self.settings.api_key.get_secret_value()}',
            'Content-Type': 'application/json',
        }
        return {
            'url': self.settings.api_base + 'text_to_speech',
            'json': json_data,
            'headers': headers,
            'params': {'GroupId': self.settings.group_id},
        }

    def generate(self, prompt: str, **kwargs: Unpack[MinimaxSpeechParametersDict]) -> TextToSpeechOutput:
        parameters = self.parameters.clone_with_changes(**kwargs)
        request_parameters = self._get_request_parameters(prompt, parameters)
        response = self.http_client.post(request_parameters=request_parameters)
        return TextToSpeechOutput(
            model_info=self.model_info,
            audio=response.content,
            audio_format='mp3',
            cost=self.calculate_cost(prompt),
        )

    async def async_generate(self, prompt: str, **kwargs: Unpack[MinimaxSpeechParametersDict]) -> TextToSpeechOutput:
        parameters = self.parameters.clone_with_changes(**kwargs)
        request_parameters = self._get_request_parameters(prompt, parameters)
        response = await self.http_client.async_post(request_parameters=request_parameters)
        return TextToSpeechOutput(
            model_info=self.model_info,
            audio=response.content,
            audio_format='mp3',
            cost=self.calculate_cost(prompt),
        )

    @property
    @override
    def name(self) -> str:
        return self.model

    @classmethod
    @override
    def from_name(cls, name: str) -> Self:
        return cls(model=name)

    @staticmethod
    def calculate_cost(text: str) -> float:
        character_count = sum(2 if '\u4e00' <= char <= '\u9fff' else 1 for char in text)
        return character_count / 1000


class MinimaxProSpeech(RemoteTextToSpeechModel):
    model_type = 'minimax_pro'

    parameters: MinimaxProSpeechParameters
    settings: MinimaxSettings

    def __init__(
        self,
        model: str = 'speech-01',
        parameters: MinimaxProSpeechParameters | None = None,
        settings: MinimaxSettings | None = None,
        http_client: HttpClient | None = None,
    ) -> None:
        parameters = parameters or MinimaxProSpeechParameters()
        settings = settings or MinimaxSettings()  # type: ignore
        http_client = http_client or HttpClient()
        super().__init__(parameters=parameters, settings=settings, http_client=http_client)

        self.model = model

    def _get_request_parameters(self, text: str, parameters: MinimaxProSpeechParameters) -> HttpxPostKwargs:
        json_data = {
            'model': self.model,
            'text': text,
            **parameters.custom_model_dump(),
        }
        headers = {
            'Authorization': f'Bearer {self.settings.api_key.get_secret_value()}',
            'Content-Type': 'application/json',
        }
        return {
            'url': self.settings.api_base + 't2a_pro',
            'json': json_data,
            'headers': headers,
            'params': {'GroupId': self.settings.group_id},
        }

    @override
    def generate(self, prompt: str, **kwargs: Unpack[MinimaxProSpeechParametersDict]) -> TextToSpeechOutput:
        parameters = self.parameters.clone_with_changes(**kwargs)
        request_parameters = self._get_request_parameters(prompt, parameters)
        response = self.http_client.post(request_parameters=request_parameters)
        response_data = response.json()
        if response_data['base_resp']['status_code'] != 0:
            raise UnexpectedResponseError(response_data)

        model_output = TextToSpeechOutput(
            model_info=self.model_info,
            audio=self.http_client.get({'url': response_data['audio_file']}).content,
            audio_format='mp3',
            cost=response_data['extra_info']['word_count'] / 1000,
        )
        model_output.extra['subtitle'] = self.http_client.get({'url': response_data['subtitle_file']}).json()
        model_output.extra.update(response_data['extra_info'])
        return model_output

    @override
    async def async_generate(self, prompt: str, **kwargs: Unpack[MinimaxProSpeechParametersDict]) -> TextToSpeechOutput:
        parameters = self.parameters.clone_with_changes(**kwargs)
        request_parameters = self._get_request_parameters(prompt, parameters)
        response = await self.http_client.async_post(request_parameters=request_parameters)
        response_data = response.json()
        if response_data['base_resp']['status_code'] != 0:
            raise UnexpectedResponseError(response_data)

        audio = (await self.http_client.async_get({'url': response_data['audio_file']})).content
        subtitle = (await self.http_client.async_get({'url': response_data['subtitle_file']})).json()

        model_output = TextToSpeechOutput(
            model_info=self.model_info,
            audio=audio,
            audio_format='mp3',
            cost=response_data['extra_info']['word_count'] / 1000,
            extra={'subtitle': subtitle},
        )
        model_output.extra.update(response_data['extra_info'])
        return model_output

    @property
    @override
    def name(self) -> str:
        return self.model

    @classmethod
    @override
    def from_name(cls, name: str) -> Self:
        return cls(model=name)
