# app/tts/providers/openai_tts.py
import os, io, requests
from typing import Optional
from openai import OpenAI  # openai>=1.40 目安
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# OpenAI の音声名（例）
# coral / shimmer / echo / ash / alloy / ballad / verse / onyx / nova / sage / fable
DEFAULT_MODEL = "gpt-4o-mini-tts"  # 2025-08 現行の TTSモデル例

class OpenAITTSProvider:
    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model

    def synthesize(self, text: str, *, voice: str = "coral", speed: float = 1.0) -> bytes:
        # OpenAI Responses API の TTS（音声生成）エンドポイント
         # 1) まず SDK（response_format）でトライ
         try:
             resp = client.audio.speech.create(
                 model=self.model,
                 voice=voice,
                 input=text,
                 response_format="wav",   # ← ここに統一
                 # speed は SDK 差分で不安定なため送らない
             )
             return getattr(resp, "read", lambda: resp.content)()
         except Exception as e:
             print(f"[openai_tts] SDK path failed: {e}. fallback to HTTP")

         # 2) だめなら HTTP 直叩き（/v1/audio/speech）
         url = "https://api.openai.com/v1/audio/speech"
         headers = {
             "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY','')}",
             "Content-Type": "application/json",
         }
         payload = {
             "model": self.model,
             "voice": voice,
             "input": text,
             "response_format": "wav",
             # "speed": speed,  # 必要になったらコメントイン（環境対応が必要）
         }
         r = requests.post(url, headers=headers, json=payload, timeout=120)
         if r.status_code != 200:
             head = r.text[:300].replace("\n", " ")
             raise RuntimeError(f"OpenAI HTTP TTS error {r.status_code}: {head}")
         return r.content
