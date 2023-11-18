from __future__ import annotations

import asyncio

import pytest

from generate.test import get_pytest_params
from generate.text_to_speech import (
    SpeechModelRegistry,
    SpeechModels,
    TextToSpeechModel,
)


def test_model_type_is_unique() -> None:
    assert len(SpeechModels) == len(SpeechModelRegistry)


@pytest.mark.parametrize('speech_model', get_pytest_params('test_text_to_speech', SpeechModelRegistry, types='model'))
def test_speech_model(speech_model: TextToSpeechModel) -> None:
    prompt = '你好，这是一个测试用例'
    sync_output = speech_model.generate(prompt)
    async_output = asyncio.run(speech_model.async_generate(prompt))

    assert len(sync_output.audio) != 0
    assert len(async_output.audio) != 0
