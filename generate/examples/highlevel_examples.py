from generate.highlevel import (generate_image, generate_speech, generate_text,
                                load_chat_model, load_image_generation_model,
                                load_speech_model)


def example_generate_text():
    try:
        prompt = "What is the weather like today?"
        model_id = "openai"
        result = generate_text(prompt=prompt, model_id=model_id)
        print("Generated text:", result.text)
    except Exception as e:
        print("Error generating text:", str(e))

def example_generate_image():
    try:
        prompt = "A futuristic city skyline at sunset"
        model_id = "openai"
        result = generate_image(prompt=prompt, model_id=model_id)
        print("Generated image path:", result.image_path)
    except Exception as e:
        print("Error generating image:", str(e))

def example_generate_speech():
    try:
        text = "Hello, welcome to our service."
        model_id = "openai"
        result = generate_speech(text=text, model_id=model_id)
        print("Generated speech path:", result.speech_path)
    except Exception as e:
        print("Error generating speech:", str(e))

def example_load_models():
    try:
        chat_model = load_chat_model("openai")
        print("Loaded chat model:", chat_model)
        image_model = load_image_generation_model("openai")
        print("Loaded image generation model:", image_model)
        speech_model = load_speech_model("openai")
        print("Loaded speech model:", speech_model)
    except Exception as e:
        print("Error loading models:", str(e))

if __name__ == "__main__":
    example_generate_text()
    example_generate_image()
    example_generate_speech()
    example_load_models()
