from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel, model_serializer


class TextToSpeechModelOutput(BaseModel):
    speech_model_id: str
    audio: bytes
    audio_format: str
    cost: Optional[float] = None
    extra: Dict[str, Any] = {}
    debug: Dict[str, Any] = {}

    @model_serializer(mode='wrap')
    def ser_model(self, handler) -> Any:  # noqa: ANN001
        output = handler(self)
        output.pop('debug')
        return output

    def save_audio(self, path: str | Path) -> None:
        path = Path(path)
        if path.suffix != f'.{self.audio_format}':
            raise ValueError(f'path suffix {path.suffix} does not match audio format {self.audio_format}')
        with open(path, 'wb') as f:
            f.write(self.audio)
