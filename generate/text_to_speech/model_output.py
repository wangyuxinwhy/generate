from __future__ import annotations

from pathlib import Path
from typing import Optional

from generate.model import ModelOutput


class TextToSpeechOutput(ModelOutput):
    audio: bytes
    audio_format: str
    cost: Optional[float] = None

    def save_audio(self, path: str | Path) -> None:
        path = Path(path)
        if path.suffix != f'.{self.audio_format}':
            raise ValueError(f'path suffix {path.suffix} does not match audio format {self.audio_format}')
        with open(path, 'wb') as f:
            f.write(self.audio)
