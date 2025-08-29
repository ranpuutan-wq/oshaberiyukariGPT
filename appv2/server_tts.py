from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel
from .tts.manager import TTSManager

app = FastAPI(title="OshiTalk TTS API")
mgr = TTSManager()

class SpeakReq(BaseModel):
    character_id: str  # "yukari" / "orig_a" など
    text: str
    style: str | None = None
    speed: float = 1.0
    pitch: float = 0.0
    emotion: str | None = None

@app.post("/speak")
async def speak(req: SpeakReq):
    wav = mgr.synth_for(req.character_id, req.text, style=req.style, speed=req.speed,
                        pitch=req.pitch, emotion=req.emotion)
    return Response(content=wav, media_type="audio/wav")
