from __future__ import annotations

from typing import Any, List, Optional, cast

from pydantic import BaseModel

from generate.chat_completion.models.dashscope import DashScopeMultiModalChat

try:
    import chainlit as cl
    import typer
    from chainlit.element import Avatar
    from chainlit.input_widget import Select, Slider, TextInput
except ImportError as e:
    raise ImportError('Please install chainlit with "pip install chainlit typer"') from e

from generate import ChatCompletionModel, load_chat_model
from generate.chat_completion.message import (
    ImagePart,
    ImageUrl,
    ImageUrlPart,
    Messages,
    SystemMessage,
    TextPart,
    UserMessage,
    UserMultiPartMessage,
)


class UserState(BaseModel):
    chat_model_id: str = 'openai'
    temperature: float = 1.0
    system_message: str = ''
    max_tokens: Optional[int] = None
    _chat_history: Messages = []

    @property
    def chat_history(self) -> Messages:
        if self.system_message:
            return [SystemMessage(content=self.system_message)] + self._chat_history
        return self._chat_history

    @property
    def chat_model(self) -> ChatCompletionModel:
        return load_chat_model(model_id=self.chat_model_id)


def get_avatars() -> List[Avatar]:
    avatar_map = {
        'openai': 'https://mrvian.com/wp-content/uploads/2023/02/logo-open-ai.png',
        'wenxin': 'https://yuxin-wang.oss-cn-beijing.aliyuncs.com/uPic/wzkPgl.png',
        'bailian': 'https://yuxin-wang.oss-cn-beijing.aliyuncs.com/uPic/kHZBrw.png',
        'dashscope_multimodal': 'https://yuxin-wang.oss-cn-beijing.aliyuncs.com/uPic/kHZBrw.png',
        'dashscope': 'https://yuxin-wang.oss-cn-beijing.aliyuncs.com/uPic/kHZBrw.png',
        'hunyuan': 'https://cdn-portal.hunyuan.tencent.com/public/static/logo/logo.png',
        'minimax': 'https://yuxin-wang.oss-cn-beijing.aliyuncs.com/uPic/lvMJ2T.png',
        'minimax_pro': 'https://yuxin-wang.oss-cn-beijing.aliyuncs.com/uPic/lvMJ2T.png',
        'zhipu': 'https://yuxin-wang.oss-cn-beijing.aliyuncs.com/uPic/HIntEu.png',
        'baichuan': 'https://yuxin-wang.oss-cn-beijing.aliyuncs.com/uPic/fODcq1.png',
        'moonshot': 'https://yuxin-wang.oss-cn-beijing.aliyuncs.com/uPic/hc2Ygt.png',
    }
    return [Avatar(name=k, url=v) for k, v in avatar_map.items()]


def get_generate_settings() -> List[Any]:
    model_select = Select(
        id='Model',
        label='Model',
        values=[
            'openai',
            'openai/gpt-4-vision-preview',
            'dashscope',
            'dashscope_multimodal',
            'zhipu',
            'zhipu/glm-4v',
            'wenxin',
            'baichuan',
            'minimax_pro',
            'moonshot',
        ],
    )
    model_id = TextInput(
        id='ModelId',
        label='Model ID',
        initial='',
        description='如 openai/gpt-4-turbo-preview，此设置会覆盖 Model 选项。',
    )
    system_message_input = TextInput(
        id='SystemMessage',
        label='System Message',
        initial='',
    )
    temperature_slider = Slider(id='Temperature', label='Temperature', min=0, max=1.0, step=0.1, initial=1)
    max_tokens = Slider(id='MaxTokens', label='Max Tokens', min=1, max=5000, step=100, initial=0)
    return [model_select, model_id, system_message_input, temperature_slider, max_tokens]


@cl.on_chat_start
async def chat_start() -> None:
    await cl.ChatSettings(get_generate_settings()).send()
    for avatar in get_avatars():
        await avatar.send()
    state = UserState()
    cl.user_session.set('state', state)


@cl.on_settings_update
async def settings_update(settings: dict) -> None:
    state = cast(UserState, cl.user_session.get('state'))
    if settings['ModelId']:
        state.chat_model_id = settings['ModelId']
    else:
        state.chat_model_id = settings['Model']
    state.temperature = settings['Temperature']
    state.system_message = settings['SystemMessage']
    if settings['MaxTokens']:
        state.max_tokens = settings['MaxTokens']


@cl.on_message
async def main(message: cl.Message) -> None:
    state = cast(UserState, cl.user_session.get('state'))
    assistant_message = cl.Message('', author=state.chat_model.model_type)
    await assistant_message.send()
    try:
        image_parts = []
        for element in message.elements:
            mime = element.mime or ''
            if mime.startswith('image'):
                image_format = mime.split('/')[1]
                if element.path is not None:
                    with open(element.path, 'rb') as image_file:
                        image_content = image_file.read()
                    if isinstance(state.chat_model, DashScopeMultiModalChat):
                        url = state.chat_model.upload_image(image_content, image_format)
                        image_url = ImageUrl(url=url)
                        image_part = ImageUrlPart(image_url=image_url)
                    else:
                        image_part = ImagePart(
                            image=image_content,
                            image_format=image_format,
                        )
                    image_parts.append(image_part)
                elif element.url is not None:
                    image_url = ImageUrl(url=element.url)
                    image_url_part = ImageUrlPart(image_url=image_url)
                    image_parts.append(image_url_part)

        if image_parts:
            text_part = TextPart(text=message.content)
            user_message = UserMultiPartMessage(content=image_parts + [text_part])
        else:
            user_message = UserMessage(content=message.content)

        state._chat_history.append(user_message)
        async for chunk in state.chat_model.async_stream_generate(
            prompt=state.chat_history,
            temperature=state.temperature,  # type: ignore
            max_tokens=state.max_tokens,  # type: ignore
        ):
            for token in chunk.stream.delta:
                await assistant_message.stream_token(token)
                await cl.sleep(0.02)  # type: ignore
            if chunk.is_finish:
                state._chat_history.append(chunk.message)
    except Exception as e:
        await cl.Message(content=f'Error: {e}').send()
    await assistant_message.update()


def chainlit_start(host: Optional[str] = None, port: Optional[int] = None) -> None:
    import os

    from chainlit.cli import run_chainlit
    from chainlit.config import DEFAULT_HOST, DEFAULT_PORT, config

    os.environ['CHAINLIT_HOST'] = host or DEFAULT_HOST
    os.environ['CHAINLIT_PORT'] = str(port or DEFAULT_PORT)
    config.project.enable_telemetry = False
    config.run.watch = False

    run_chainlit(__file__)


if __name__ == '__main__':
    typer.run(chainlit_start)
