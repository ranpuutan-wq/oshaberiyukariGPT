# app/server_tts.py
from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel
from app.tts.manager import TTSManager   # 新しい
import os
from fastapi.responses import JSONResponse

from app.tts.cevio_com_dyn import cevio_available_casts
from contextlib import asynccontextmanager
from fastapi import FastAPI
import logging


mgr = TTSManager()

for spk in ("yukari","maki","ia","one"):
    try:
        print("[route]", spk, mgr.describe_voice(spk))
    except Exception as e:
        print("[route][WARN]", spk, e)

app = FastAPI()


# ---------- モデル ----------
class SpeakReq(BaseModel):
    speaker: str  # "yukari" / "maki" / "ia" / "one"
    text: str
    style: str | None = None
    speed: float = 1.0
    pitch: float = 0.0
    emotion: str | None = None

@app.get("/health")
def health():
    return {"ok": True}

# ★ここだけに統一：/speak（実URLは /tts/speak）
@app.post("/speak")
async def speak(req: SpeakReq):
    try:
        wav = await mgr.synth_for_async(
            req.speaker,
            req.text,
            speed=req.speed,
            pitch=req.pitch,
            style=req.style,
            emotion=req.emotion,
        )
        return Response(content=wav, media_type="audio/wav")
    except Exception as e:
        # フォールバックはしない方針 → 明示エラー
        return JSONResponse({"error": str(e)}, status_code=503)