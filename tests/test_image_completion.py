from generate import OpenAIChat


def test_image_cimpletion() -> None:
    user_input = {
        'role': 'user',
        'content': [
            {'type': 'text', 'text': 'What’s in this image?'},
            {
                'type': 'image_url',
                'image_url': {
                    'url': 'https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg',
                    'detail': 'high',
                },
            },
        ],
    }
    chat = OpenAIChat(model='gpt-4-vision-preview')
    output = chat.generate(user_input, max_tokens=10)
    assert output.reply != ''
