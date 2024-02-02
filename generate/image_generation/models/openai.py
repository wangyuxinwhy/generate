from __future__ import annotations

import base64
from typing import Literal, Optional

from httpx import Response
from pydantic import Field
from typing_extensions import Annotated, Self, Unpack, override

from generate.http import HttpClient, HttpxPostKwargs
from generate.image_generation.base import GeneratedImage, ImageGenerationOutput, RemoteImageGenerationModel
from generate.model import ModelParameters, ModelParametersDict
from generate.platforms.openai import OpenAISettings

MAX_PROMPT_LENGTH_DALLE_3 = 4000
MAX_PROMPT_LENGTH_DALLE_2 = 1000
OPENAI_IMAGE_GENERATION_PRICE_MAP = {
    'dall-e-3': {
        'hd': {
            '1024x1024': 0.04,
            '1792x1024': 0.08,
            '1024x1792': 0.08,
        },
        'standard': {
            '1024x1024': 0.08,
            '1792x1024': 0.12,
            '1024x1792': 0.12,
        },
    },
    'dall-e-2': {
        'standard': {
            '256x256': 0.016,
            '512x512': 0.018,
            '1024x1024': 0.02,
        }
    },
}


class OpenAIImageGenerationParameters(ModelParameters):
    quality: Optional[Literal['hd', 'standard']] = None
    response_format: Optional[Literal['url', 'b64_json']] = None
    size: Optional[Literal['256x256', '512x512', '1024x1024', '1792x1024', '1024x1792']] = None
    style: Optional[Literal['vivid', 'natural']] = None
    n: Optional[Annotated[int, Field(ge=1, le=10)]] = None
    user: Optional[str] = None


class OpenAIImageGenerationParametersDict(ModelParametersDict, total=False):
    quality: Optional[Literal['hd', 'standard']]
    response_format: Optional[Literal['url', 'b64_json']]
    size: Optional[Literal['256x256', '512x512', '1024x1024', '1792x1024', '1024x1792']]
    style: Optional[Literal['vivid', 'natural']]
    n: Optional[int]
    user: Optional[str]


class OpenAIImageGeneration(RemoteImageGenerationModel):
    model_type = 'openai'

    parameters: OpenAIImageGenerationParameters
    settings: OpenAISettings

    def __init__(
        self,
        model: str = 'dall-e-3',
        parameters: OpenAIImageGenerationParameters | None = None,
        settings: OpenAISettings | None = None,
        http_client: HttpClient | None = None,
    ) -> None:
        parameters = parameters or OpenAIImageGenerationParameters()
        settings = settings or OpenAISettings()  # type: ignore
        http_client = http_client or HttpClient()
        super().__init__(parameters=parameters, settings=settings, http_client=http_client)

        self.model = model
        self._check_parameters()

    def _check_parameters(self) -> None:
        if self.model == 'dall-e-3':
            if self.parameters.n is not None and self.parameters.n != 1:
                raise ValueError('dall-e-3 only supports n=1')
            size = self.parameters.size
            if size is not None and size not in ('1024x1024', '1792x1024', '1024x1792'):
                raise ValueError('dall-e-3 only supports size=1024x1024, 1792x1024, 1024x1792')
        if self.model == 'dall-e-2':
            if self.parameters.quality is not None:
                raise ValueError('dall-e-2 does not support quality')
            size = self.parameters.size
            if size is not None and size not in ('256x256', '512x512', '1024x1024'):
                raise ValueError('dall-e-2 only supports size=256x256, 512x512, 1024x1024')
            if self.parameters.style is not None:
                raise ValueError('dall-e-2 does not support style')

    def _check_prompt(self, prompt: str) -> None:
        if self.model == 'dall-e-3' and len(prompt) >= MAX_PROMPT_LENGTH_DALLE_3:
            raise ValueError('dall-e-3 does not support prompt length >= 4000')

        if self.model == 'dall-e-2' and len(prompt) >= MAX_PROMPT_LENGTH_DALLE_2:
            raise ValueError('dall-e-2 does not support prompt length >= 100')

    def _get_request_parameters(self, prompt: str, parameters: OpenAIImageGenerationParameters) -> HttpxPostKwargs:
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.settings.api_key.get_secret_value()}',
        }
        json_data = {
            'model': self.model,
            'prompt': prompt,
            **parameters.custom_model_dump(),
        }
        return {
            'url': self.settings.api_base + 'images/generations',
            'json': json_data,
            'headers': headers,
        }

    @override
    def generate(self, prompt: str, **kwargs: Unpack[OpenAIImageGenerationParametersDict]) -> ImageGenerationOutput:
        self._check_prompt(prompt)
        parameters = self.parameters.clone_with_changes(**kwargs)
        request_parameters = self._get_request_parameters(prompt, parameters)
        response = self.http_client.post(request_parameters=request_parameters)
        return self._construct_model_output(prompt, parameters, response)

    @override
    async def async_generate(self, prompt: str, **kwargs: Unpack[OpenAIImageGenerationParametersDict]) -> ImageGenerationOutput:
        self._check_prompt(prompt)
        parameters = self.parameters.clone_with_changes(**kwargs)
        request_parameters = self._get_request_parameters(prompt, parameters)
        response = await self.http_client.async_post(request_parameters=request_parameters)
        return self._construct_model_output(prompt, parameters, response)

    def _construct_model_output(
        self, prompt: str, parameters: OpenAIImageGenerationParameters, response: Response
    ) -> ImageGenerationOutput:
        response_data = response.json()
        generated_images: list[GeneratedImage] = []
        for image_data in response_data['data']:
            image_prompt = image_data.get('revised_prompt') or prompt
            url = image_data.get('url')
            if url:
                content = self.http_client.get({'url': url}).content
            else:
                b64 = image_data.get('b64_json')
                if b64 is None:
                    raise ValueError('No URL or b64_json found in response')
                content = base64.b64decode(b64)
            generated_images.append(
                GeneratedImage(
                    url=url,
                    prompt=image_prompt,
                    image_format='png',
                    content=content,
                )
            )
        return ImageGenerationOutput(
            model_info=self.model_info,
            images=generated_images,
            cost=self.calculate_cost(parameters),
        )

    def calculate_cost(self, parameters: OpenAIImageGenerationParameters) -> float:
        dollar_to_yuan = 7
        quality = parameters.quality or 'standard'
        size = parameters.size or '1024x1024'
        model_price = OPENAI_IMAGE_GENERATION_PRICE_MAP[self.model][quality][size]
        return model_price * dollar_to_yuan

    @property
    @override
    def name(self) -> str:
        return self.model

    @classmethod
    @override
    def from_name(cls, name: str) -> Self:
        return cls(model=name)
