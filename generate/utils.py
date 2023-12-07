from generate.chat_completion import ChatCompletionModel, ChatCompletionOutput, ChatModelRegistry
from generate.chat_completion.message import Prompt
from generate.image_generation import ImageGenerationModel, ImageGenerationModelRegistry, ImageGenerationOutput
from generate.text_to_speech import SpeechModelRegistry, TextToSpeechModel, TextToSpeechOutput


def load_chat_model(model_id: str) -> ChatCompletionModel:
    if '/' not in model_id:
        model_type = model_id
        return ChatModelRegistry[model_type][0]()
    model_type, name = model_id.split('/')
    model_cls = ChatModelRegistry[model_type][0]
    return model_cls.from_name(name)


def load_speech_model(model_id: str) -> TextToSpeechModel:
    if '/' not in model_id:
        model_type = model_id
        return SpeechModelRegistry[model_type][0]()
    model_type, name = model_id.split('/')
    model_cls = SpeechModelRegistry[model_type][0]
    return model_cls.from_name(name)


def load_image_generation_model(model_id: str) -> ImageGenerationModel:
    if '/' not in model_id:
        model_type = model_id
        return ImageGenerationModelRegistry[model_type][0]()
    model_type, name = model_id.split('/')
    model_cls = ImageGenerationModelRegistry[model_type][0]
    return model_cls.from_name(name)


def generate_text(prompt: Prompt, model_id: str = 'openai') -> ChatCompletionOutput:
    model = load_chat_model(model_id)
    return model.generate(prompt)


def generate_speech(text: str, model_id: str = 'openai') -> TextToSpeechOutput:
    model = load_speech_model(model_id)
    return model.generate(text)


def generate_image(prompt: str, model_id: str = 'openai') -> ImageGenerationOutput:
    model = load_image_generation_model(model_id)
    return model.generate(prompt)
