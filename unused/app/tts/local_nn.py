import requests
from .base import TTSAdapter, TTSRequest

class LocalNeuralAdapter(TTSAdapter):
    def __init__(self, base="http://127.0.0.1:8020"):  # 自前TTSサーバ等
        self.base = base
    def synth(self, req: TTSRequest) -> bytes:
        r = requests.post(f"{self.base}/synth", json={
            "text": req.text,
            "speaker": req.speaker, # オリジナルボイスID
            "speed": req.speed,
            "emotion": req.emotion
        }, timeout=60)
        r.raise_for_status()
        return r.content
