from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

@dataclass
class TTSRequest:
    text: str
    speaker: str
    style: Optional[str] = None   # "friendly", "whisper", etc.
    speed: float = 1.0
    pitch: float = 0.0
    emotion: Optional[str] = None # "joy","sad","calm" など

class TTSAdapter(ABC):
    @abstractmethod
    def synth(self, req: TTSRequest) -> bytes:  # returns WAV bytes
        ...
