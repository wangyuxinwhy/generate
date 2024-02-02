from __future__ import annotations

from httpx import Response
from typing_extensions import Self, override

from generate.http import HttpClient, HttpxPostKwargs
from generate.image_generation.base import GeneratedImage, ImageGenerationOutput, RemoteImageGenerationModel
from generate.model import ModelParameters
from generate.platforms.zhipu import ZhipuSettings, generate_zhipu_token


class ZhipuImageGenerationParameters(ModelParameters):
    pass


class ZhipuImageGeneration(RemoteImageGenerationModel):
    model_type = 'zhipu'

    parameters: ZhipuImageGenerationParameters
    settings: ZhipuSettings

    def __init__(
        self,
        model: str = 'cogview-3',
        settings: ZhipuSettings | None = None,
        http_client: HttpClient | None = None,
    ) -> None:
        parameters = ZhipuImageGenerationParameters()
        settings = settings or ZhipuSettings()  # type: ignore
        http_client = http_client or HttpClient()
        super().__init__(parameters=parameters, settings=settings, http_client=http_client)

        self.model = model

    def _get_request_parameters(self, prompt: str) -> HttpxPostKwargs:
        headers = {
            'Content-Type': 'application/json',
            'Authorization': generate_zhipu_token(api_key=self.settings.api_key.get_secret_value()),
        }
        json_data = {
            'model': self.model,
            'prompt': prompt,
        }
        return {
            'url': self.settings.v4_api_base + 'images/generations',
            'json': json_data,
            'headers': headers,
        }

    @override
    def generate(self, prompt: str) -> ImageGenerationOutput:
        request_parameters = self._get_request_parameters(prompt)
        response = self.http_client.post(request_parameters=request_parameters)
        return self._construct_model_output(prompt, response)

    @override
    async def async_generate(self, prompt: str) -> ImageGenerationOutput:
        request_parameters = self._get_request_parameters(prompt)
        response = await self.http_client.async_post(request_parameters=request_parameters)
        return self._construct_model_output(prompt, response)

    def _construct_model_output(self, prompt: str, response: Response) -> ImageGenerationOutput:
        response_data = response.json()
        generated_images: list[GeneratedImage] = []
        for image_data in response_data['data']:
            url = image_data['url']
            content = self.http_client.get({'url': url}).content
            generated_images.append(
                GeneratedImage(
                    url=url,
                    prompt=prompt,
                    image_format='png',
                    content=content,
                )
            )
        return ImageGenerationOutput(
            model_info=self.model_info,
            images=generated_images,
            cost=0.25,
        )

    @property
    @override
    def name(self) -> str:
        return self.model

    @classmethod
    @override
    def from_name(cls, name: str) -> Self:
        return cls(model=name)
