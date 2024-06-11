from generate.highlevel import generate_text, load_chat_model


def test_load_chat_model() -> None:
    model = load_chat_model('openai/gpt-3.5-turbo')
    assert model.model_type == 'openai'
    assert model.name == 'gpt-3.5-turbo'


def test_generate_text() -> None:
    output = generate_text('你好')
    assert output.reply != ''
