from __future__ import annotations

import os
from typing import Any, ClassVar, Literal, Optional

from httpx import Response
from pydantic import Field
from typing_extensions import Annotated, Self, Unpack, override

from generate.http import HttpModelInitKwargs, HttpxPostKwargs
from generate.parameters import ModelParameters
from generate.text_to_speech.http_speech import HttpSpeechModel
from generate.text_to_speech.model_output import TextToSpeechModelOutput


class OpenAISpeechParameters(ModelParameters):
    voice_id: Annotated[str, Field(serialization_alias='voice')] = 'alloy'
    response_format: Optional[Literal['mp3', 'aac', 'opus', 'flac']] = None
    speed: Annotated[Optional[float], Field(ge=0.25, le=4.0)] = None


class OpenAISpeech(HttpSpeechModel[OpenAISpeechParameters]):
    model_type = 'openai'
    default_api_base: ClassVar[str] = 'https://api.openai.com/v1/audio/speech'

    def __init__(
        self,
        model: str = 'tts-1',
        api_key: str | None = None,
        api_base: str | None = None,
        parameters: OpenAISpeechParameters | None = None,
        **kwargs: Unpack[HttpModelInitKwargs],
    ) -> None:
        parameters = parameters or OpenAISpeechParameters()
        super().__init__(parameters, **kwargs)
        self.model = model
        self.api_base = api_base or os.getenv('OPENAI_API_BASE') or self.default_api_base
        self.api_key = api_key or os.environ['OPENAI_API_KEY']

    @override
    def _get_request_parameters(self, text: str, parameters: OpenAISpeechParameters) -> HttpxPostKwargs:
        parameters_dict = parameters.model_dump(exclude_none=True, by_alias=True)
        json_data = {
            'model': self.model,
            'input': text,
            **parameters_dict,
        }
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
        }
        return {
            'url': self.api_base,
            'json': json_data,
            'headers': headers,
        }

    @override
    def _construct_model_output(
        self, text: str, parameters: OpenAISpeechParameters, response: Response
    ) -> TextToSpeechModelOutput:
        return TextToSpeechModelOutput(
            speech_model_id=self.model_id,
            audio=response.content,
            audio_format=parameters.response_format or 'mp3',
            cost=self.calculate_cost(text),
        )

    @property
    @override
    def name(self) -> str:
        return self.model

    @classmethod
    @override
    def from_name(cls, name: str, **kwargs: Any) -> Self:
        return cls(model=name, **kwargs)

    def calculate_cost(self, text: str) -> float | None:
        dollar_to_yuan = 7
        if self.model == 'tts-1':
            return (len(text) / 1000) * (0.015 * dollar_to_yuan)

        if self.model == 'tts-1-hd':
            return (len(text) / 1000) * (0.03 * dollar_to_yuan)

        return None
