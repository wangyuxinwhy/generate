from __future__ import annotations

import asyncio
import time
from typing import Any, Literal, Optional, Union

from pydantic import Base64Str, Field, HttpUrl
from typing_extensions import Annotated, Self

from generate.http import HttpClient, HttpxPostKwargs, UnexpectedResponseError
from generate.image_generation.base import GeneratedImage, ImageGenerationModel, ImageGenerationOutput
from generate.model import ModelParameters
from generate.platforms.baidu import BaiduCreationSettings
from generate.token import TokenMixin

ValidSize = Literal[
    '512x512',
    '640x360',
    '360x640',
    '1024x1024',
    '1280x720',
    '720x1280',
    '2048x2048',
    '2560x1440',
    '1440x2560',
]


class BaiduImageGenerationParameters(ModelParameters):
    size: ValidSize = '1024x1024'
    n: Optional[Annotated[int, Field(ge=1, le=8)]] = None
    reference_image: Union[HttpUrl, Base64Str, None] = None
    change_degree: Optional[Annotated[int, Field(ge=1, le=10)]] = None

    def custom_model_dump(self) -> dict[str, Any]:
        output_data: dict[str, Any] = {}
        width, height = self.size.split('x')
        output_data['width'] = int(width)
        output_data['height'] = int(height)
        n = self.n or 1
        output_data['image_num'] = n
        if self.reference_image:
            if isinstance(self.reference_image, HttpUrl):
                output_data['url'] = self.reference_image
            else:
                output_data['image'] = self.reference_image
        if self.change_degree:
            output_data['change_degree'] = self.change_degree
        return output_data


class BaiduImageGeneration(ImageGenerationModel[BaiduImageGenerationParameters], TokenMixin):
    model_type = 'baidu'

    def __init__(
        self,
        settings: BaiduCreationSettings | None = None,
        parameters: BaiduImageGenerationParameters | None = None,
        http_client: HttpClient | None = None,
    ) -> None:
        parameters = parameters or BaiduImageGenerationParameters()
        super().__init__(parameters)

        self.settings = settings or BaiduCreationSettings()  # type: ignore
        self.http_client = http_client or HttpClient()
        self.task_timeout = 60

    def _get_token(self) -> str:
        headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
        params = {
            'grant_type': 'client_credentials',
            'client_id': self.settings.api_key.get_secret_value(),
            'client_secret': self.settings.secret_key.get_secret_value(),
        }
        response = self.http_client.post(
            {
                'url': self.settings.access_token_api,
                'headers': headers,
                'params': params,
                'json': None,
            }
        )
        response_dict = response.json()
        if 'error' in response_dict:
            raise UnexpectedResponseError(response_dict)
        return response_dict['access_token']

    def _get_request_parameters(self, prompt: str, parameters: BaiduImageGenerationParameters) -> HttpxPostKwargs:
        headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
        json_data = {
            'prompt': prompt,
            **parameters.custom_model_dump(),
        }
        return {
            'url': self.settings.image_generation_api,
            'json': json_data,
            'headers': headers,
            'params': {
                'access_token': self.token,
            },
        }

    def _image_generation(self, prompt: str, parameters: BaiduImageGenerationParameters) -> ImageGenerationOutput:
        request_parameters = self._get_request_parameters(prompt, parameters)
        response = self.http_client.post(request_parameters=request_parameters)
        image_urls: list[str] = self._get_image(response.json()['data']['task_id'])
        images: list[GeneratedImage] = []
        for image_url in image_urls:
            image = GeneratedImage(
                url=image_url,
                prompt=prompt,
                image_format='png',
                content=self.http_client.get({'url': image_url}).content,
            )
            images.append(image)
        return ImageGenerationOutput(model_info=self.model_info, cost=0.3 * (parameters.n or 1), images=images)

    async def _async_image_generation(self, prompt: str, parameters: BaiduImageGenerationParameters) -> ImageGenerationOutput:
        request_parameters = self._get_request_parameters(prompt, parameters)
        response = await self.http_client.async_post(request_parameters=request_parameters)
        image_urls: list[str] = await self._async_get_image(response.json()['data']['task_id'])
        images: list[GeneratedImage] = []
        images_response = await asyncio.gather(*[self.http_client.async_get({'url': image_url}) for image_url in image_urls])
        for image_url, image_response in zip(image_urls, images_response):
            image = GeneratedImage(
                url=image_url,
                prompt=prompt,
                image_format='png',
                content=image_response.content,
            )
            images.append(image)
        return ImageGenerationOutput(model_info=self.model_info, cost=0.3 * (parameters.n or 1), images=images)

    def _get_image_request_parameters(self, task_id: str) -> HttpxPostKwargs:
        return {
            'url': 'https://aip.baidubce.com/rpc/2.0/ernievilg/v1/getImgv2',
            'params': {
                'access_token': self.token,
            },
            'headers': {'Content-Type': 'application/json'},
            'json': {'task_id': str(task_id)},
        }

    def _parse_task_info(self, task_info: dict[str, Any]) -> list[str] | None:
        task_status = task_info['data']['task_status']
        if task_status == 'FAILED':
            raise UnexpectedResponseError(task_info, 'Task failed')
        if task_status == 'SUCCESS':
            image_urls: list[str] = []
            for sub_result in task_info['data']['sub_task_result_list']:
                image_url = sub_result['final_image_list'][0]['img_url']
                image_urls.append(image_url)
            return image_urls
        return None

    def _get_image(self, task_id: str) -> list[str]:
        start_time = time.time()
        task_info: dict[str, Any] = {}
        while (time.time() - start_time) < self.task_timeout:
            response = self.http_client.post(self._get_image_request_parameters(task_id))
            task_info = response.json()
            image_urls = self._parse_task_info(task_info)
            if image_urls:
                return image_urls
            time.sleep(1)
        raise UnexpectedResponseError(task_info, 'Timeout')

    async def _async_get_image(self, task_id: str) -> list[str]:
        start_time = time.time()
        task_info: dict[str, Any] = {}
        while (time.time() - start_time) < self.task_timeout:
            response = await self.http_client.async_post(self._get_image_request_parameters(task_id))
            task_info = response.json()
            image_urls = self._parse_task_info(task_info)
            if image_urls:
                return image_urls
            await asyncio.sleep(1)
        raise UnexpectedResponseError(task_info, 'Timeout')

    @property
    def name(self) -> str:
        return 'getImgv2'

    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> Self:
        if name and name != 'getImgv2':
            raise ValueError(f'Invalid model name: {name}, expected: getImgv2')
        return cls(**kwargs)
