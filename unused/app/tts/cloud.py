import requests, os
from .base import TTSAdapter, TTSRequest

class CloudAdapter(TTSAdapter):
    def __init__(self, base_url:str, api_key:str, voice_id:str):
        self.base = base_url
        self.key = api_key
        self.voice = voice_id
    def synth(self, req: TTSRequest) -> bytes:
        # 例: 任意のTTSベンダAPI（SSMLやstyle指定を適宜対応）
        headers = {"Authorization": f"Bearer {self.key}"}
        data = {
            "text": req.text,
            "voice": self.voice,
            "style": req.style or "casual",
            "speed": req.speed
        }
        r = requests.post(f"{self.base}/tts", json=data, headers=headers, timeout=60)
        r.raise_for_status()
        return r.content
