from generate.utils import (
    generate_image,
    generate_speech,
    generate_text,
    load_chat_model,
    load_image_generation_model,
    load_speech_model,
)


def test_load_chat_model() -> None:
    model = load_chat_model('openai/gpt-3.5-turbo')
    assert model.model_type == 'openai'
    assert model.name == 'gpt-3.5-turbo'


def test_load_speech_model() -> None:
    model = load_speech_model('openai/tts-1-hd')
    assert model.model_type == 'openai'
    assert model.name == 'tts-1-hd'


def test_load_image_generation_model() -> None:
    model = load_image_generation_model('openai/dall-e-3')
    assert model.model_type == 'openai'
    assert model.name == 'dall-e-3'


def test_generate_text() -> None:
    output = generate_text('你好')
    assert output.reply != ''


def test_generate_speech() -> None:
    output = generate_speech('这是一个测试用例')
    assert len(output.audio) != 0


def test_generate_image() -> None:
    output = generate_image('可爱的猫', model_id='openai/dall-e-2')
    assert len(output.images[0].content) != 0
