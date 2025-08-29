from .base import TTSRequest
from .aivoice import AIVoiceAdapter
from .cloud import CloudAdapter
from .local_nn import LocalNeuralAdapter

class TTSManager:
    def __init__(self):
        self.aivoice = AIVoiceAdapter()
        self.cloud_hi = CloudAdapter(
            base_url="https://api.example-tts.com", api_key="YOUR_KEY", voice_id="origA"
        )
        self.local_nn = LocalNeuralAdapter()
    def synth_for(self, character_id: str, text: str, **opts) -> bytes:
        req = TTSRequest(text=text, speaker=character_id, **opts)
        if character_id == "yukari":
            return self.aivoice.synth(req)
        elif character_id.startswith("orig_"):
            # まずクラウド高品質、失敗したらローカルへ
            try:
                return self.cloud_hi.synth(req)
            except Exception:
                return self.local_nn.synth(req)
        else:
            # デフォルトフォールバック（VOICEVOX等を別途実装も可）
            return self.local_nn.synth(req)
