import requests
from .base import TTSAdapter, TTSRequest

class AIVoiceAdapter(TTSAdapter):
    def __init__(self, base="http://127.0.0.1:50031"):
        self.base = base
    def synth(self, req: TTSRequest) -> bytes:
        payload = {
            "text": req.text,
            "speaker": "yukari",  # ゆかり固定 or マッピング
            "params": {
                "speedScale": req.speed,
                "pitchScale": req.pitch,
                "intonationScale": 1.0
            }
        }
        r = requests.post(f"{self.base}/v1/tts", json=payload, timeout=60)
        r.raise_for_status()
        return r.content
