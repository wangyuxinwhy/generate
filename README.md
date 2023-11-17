# Generate

> One API to Access World-Class Generative Models.

## 简介

Generate Package 允许用户通过统一的 api 访问跨平台的生成式模型，当前支持：

* [OpenAI](https://platform.openai.com/docs/introduction)
* [Azure](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/chatgpt?tabs=python&amp;pivots=programming-language-chat-completions)
* [阿里云-百炼](https://bailian.console.aliyun.com/)
* [百川智能](https://platform.baichuan-ai.com/docs/api)
* [腾讯云-混元](https://cloud.tencent.com/document/product/1729)
* [Minimax](https://api.minimax.chat/document/guides/chat)
* [百度智能云](https://cloud.baidu.com/doc/WENXINWORKSHOP/s/clntwmv7t)
* [智谱](https://open.bigmodel.cn/dev/api)
* ...

## Features

* **多模态**，支持文本生成，图像生成以及语音生成
* **跨平台**，完整支持 OpenAI，Azure，Minimax 在内的多家平台，
* **One API**，统一了不同平台的消息格式，推理参数，接口封装，返回解析
* **异步和流式**，提供流式调用，非流式调用，同步调用，异步调用，适配不同的应用场景
* **自带电池**，提供输入检查，参数检查，计费，*ChatEngine*, *CompletionEngine*, *function* 等功能
* **高质量代码**，100% typehints，pylance strict, ruff lint & format,  test coverage > 85% ...

> 完整支持是指，只要是平台提供的功能和参数，`generate` 包都原生支持，不打折扣！比如，OpenAI 的 Function Call, Tool Calls，MinimaxPro 的 Plugins 等
## 基础使用

<a target="_blank" href="https://colab.research.google.com/github/wangyuxinwhy/generate/blob/main/examples/tutorial.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

### 安装

```bash
pip install generate-core
```

### 文本生成

```python
from generate import OpenAIChat

model = OpenAIChat()
model.generate('你好，GPT！', temperature=0, seed=2023)

# ----- Output -----
ChatCompletionOutput(
    model_info=ModelInfo(task='chat_completion', type='openai', name='gpt-3.5-turbo-0613'),
    cost=0.000343,
    extra={'usage': {'prompt_tokens': 13, 'completion_tokens': 18, 'total_tokens': 31}},
    messages=[
        AssistantMessage(
            content='你好！有什么我可以帮助你的吗？',
            role='assistant',
            name=None,
            content_type='text'
        )
    ],
    finish_reason='stop'
)
```

### 图像生成

```python
from generate import OpenAIImageGeneration

model = OpenAIImageGeneration()
model.generate('black hole')

# ----- Output -----
ImageGenerationOutput(
    model_info=ModelInfo(task='image_generation', type='openai', name='dall-e-3'),
    cost=0.56,
    extra={},
    images=[
        GeneratedImage(
            url='https://oaidalleapiprodscus.blob.core.windows.net/...',
            prompt='Visualize an astronomical illustration featuring a black hole at its core. The black hole 
should be portrayed with strong gravitational lensing effect that distorts the light around it. Include a 
surrounding accretion disk, glowing brightly with blue and white hues, streaked with shades of red and orange, 
indicating heat and intense energy. The cosmos in the background should be filled with distant stars, galaxies, and
nebulas, illuminating the vast, infinite space with specks of light.',
            image_format='png',
            content=b'<image bytes>'
        )
    ]
)
```

### 语音生成

```python
from generate import MinimaxSpeech

model = MinimaxSpeech()
model.generate('你好，世界！')

# ----- Output -----
TextToSpeechOutput(
    model_info=ModelInfo(task='text_to_speech', type='minimax', name='speech-01'),
    cost=0.01,
    extra={},
    audio=b'<audio bytes>',
    audio_format='mp3'
)
```

‍