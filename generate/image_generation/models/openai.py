from __future__ import annotations

import base64
import os
from typing import Any, Literal, Optional

from httpx import Response
from pydantic import Field
from typing_extensions import Annotated, Self, Unpack

from generate.http import HttpClient, HttpClientInitKwargs, HttpxPostKwargs
from generate.image_generation.base import ImageGenerationModel
from generate.image_generation.model_output import GeneratedImage, ImageGenerationOutput
from generate.parameters import ModelParameters

MAX_PROMPT_LENGTH_DALLE_3 = 4000
MAX_PROMPT_LENGTH_DALLE_2 = 1000


class OpenAIImageGenerationParameters(ModelParameters):
    quality: Optional[Literal['hd', 'standard']] = None
    response_format: Optional[Literal['url', 'b64_json']] = None
    size: Optional[Literal['256x256', '512x512', '1024x1024', '1792x1024', '1024x1792']] = None
    style: Optional[Literal['vivid', 'natural']] = None
    n: Optional[Annotated[int, Field(ge=1, le=10)]] = None
    user: Optional[str] = None


class OpenAIImageGeneration(ImageGenerationModel[OpenAIImageGenerationParameters]):
    model_type = 'openai'
    default_api_base: str = 'https://api.openai.com/v1/images/generations'

    def __init__(
        self,
        model: str = 'dall-e-3',
        api_key: str | None = None,
        api_base: str | None = None,
        parameters: OpenAIImageGenerationParameters | None = None,
        **kwargs: Unpack[HttpClientInitKwargs],
    ) -> None:
        parameters = parameters or OpenAIImageGenerationParameters()
        super().__init__(parameters)
        self.model = model
        self.api_base = api_base or self.default_api_base
        self.api_key = api_key or os.environ['OPENAI_API_KEY']
        self.http_client = HttpClient(**kwargs)
        self.check_parameters()

    def check_parameters(self) -> None:
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

    def check_prompt(self, prompt: str) -> None:
        if self.model == 'dall-e-3' and len(prompt) >= MAX_PROMPT_LENGTH_DALLE_3:
            raise ValueError('dall-e-3 does not support prompt length >= 4000')

        if self.model == 'dall-e-2' and len(prompt) >= MAX_PROMPT_LENGTH_DALLE_2:
            raise ValueError('dall-e-2 does not support prompt length >= 100')

    def _get_request_parameters(self, prompt: str, parameters: OpenAIImageGenerationParameters) -> HttpxPostKwargs:
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}',
        }
        json_data = {
            'model': self.model,
            'prompt': prompt,
            **parameters.model_dump(exclude_none=True),
        }
        return {
            'url': self.api_base,
            'json': json_data,
            'headers': headers,
        }

    def _image_generation(self, prompt: str, parameters: OpenAIImageGenerationParameters) -> ImageGenerationOutput:
        self.check_prompt(prompt)
        request_parameters = self._get_request_parameters(prompt, parameters)
        response = self.http_client.post(request_parameters=request_parameters)
        return self._construct_model_output(prompt, parameters, response)

    async def _async_image_generation(self, prompt: str, parameters: OpenAIImageGenerationParameters) -> ImageGenerationOutput:
        self.check_prompt(prompt)
        request_parameters = self._get_request_parameters(prompt, parameters)
        response = await self.http_client.async_post(request_parameters=request_parameters)
        return self._construct_model_output(prompt, parameters, response)

    def _construct_model_output(
        self, prompt: str, parameters: OpenAIImageGenerationParameters, response: Response
    ) -> ImageGenerationOutput:
        response_data = response.json()
        generated_images = []
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
        )

    @property
    def name(self) -> str:
        return self.model

    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> Self:
        return cls(model=name, **kwargs)
