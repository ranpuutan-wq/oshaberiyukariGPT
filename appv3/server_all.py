# app/server_all.py
from __future__ import annotations
import os, asyncio, logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel

# 生成まわり
from app.llm_openai import generate_turns
# TTSまわり
from app.tts.manager import TTSManager
# CeVIO列挙（任意）
from app.tts.cevio_com_dyn import cevio_available_casts, cevio_ai_available_casts
from app.tts.aivoice_api import AIVoiceClient

log = logging.getLogger("server_all")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

# ---------- lifespan (startup/shutdown) ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.cevio_casts = {"cevio": [], "cevioai": []}
    app.state.aivoice_voices = []

    async def list_cevio_all():
        try:
            # CS と AI を両方列挙
            cs = cevio_available_casts() or []
            ai = cevio_ai_available_casts() or []
            app.state.cevio_casts = {"cevio": cs, "cevioai": ai}
            print("[startup] CeVIO casts:")
            for c in cs: print("  [cevio cs] ", c)
            print("[startup] CeVIO AI casts:")
            for c in ai: print("  [cevio ai] ", c)
        except Exception as e:
            print("[startup][WARN] CeVIO列挙失敗:", e)

    async def list_aivoice():
        # COM をイベントループ外で実行（ブロッキング対策）
        import anyio
        def work():
            cli = AIVoiceClient()
            try:
                cli.connect()
                return cli.list_voices()
            finally:
                cli.close()
        try:
            voices = await anyio.to_thread.run_sync(work)
            app.state.aivoice_voices = voices or []
            print("[startup] AIVOICE voices:")
            for v in app.state.aivoice_voices:
                print("  [aivoice] ", v)
        except Exception as e:
            print("[startup][WARN] AIVOICE列挙失敗:", e)

    # 起動時に並列で列挙
    await asyncio.gather(list_cevio_all(), list_aivoice())
    yield
    print("[server] bye.")


# ---------- FastAPI ----------
app = FastAPI(title="AI4人チャットAPI (single file)", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ---------- Routers ----------
gen = APIRouter(prefix="/gen", tags=["gen"])
tts = APIRouter(prefix="/tts", tags=["tts"])

@tts.get("/casts")
def list_all_casts():
    return {
        "cevio":   getattr(app.state, "cevio_casts", {}).get("cevio", []),
        "cevioai": getattr(app.state, "cevio_casts", {}).get("cevioai", []),
        "aivoice": getattr(app.state, "aivoice_voices", []),
    }

# ========== 生成API ==========
class GenReq(BaseModel):
    topic: str
    turns: int = 12
    model: str | None = None

@gen.post("/generate_4")
def generate_4(req: GenReq):
    model = req.model or "gpt-4.1"  # or your default
    turns = max(4, min(64, req.turns))
    turns_data = generate_turns(req.topic, turns, model=model)
    return {"turns": turns_data}

# ========== TTS API ==========
mgr = TTSManager()
# ルート確認ログ（起動時1回）
for spk in ("yukari", "maki", "ia", "one"):
    try:
        print("[route]", spk, mgr.describe_voice(spk))
    except Exception as e:
        print("[route][WARN]", spk, e)

class SpeakReq(BaseModel):
    speaker: str  # "yukari" | "maki" | "ia" | "one"
    text: str
    style: str | None = None
    speed: float = 1.0
    pitch: float = 0.0
    emotion: str | None = None

@tts.get("/health")
def tts_health():
    return {"ok": True}

@tts.post("/speak")
async def speak(req: SpeakReq):
    try:
        # タイムアウト保険（詰まり対策）
        wav = await asyncio.wait_for(
            mgr.synth_for_async(
                req.speaker, req.text,
                speed=req.speed, pitch=req.pitch,
                style=req.style, emotion=req.emotion,
            ),
            timeout=float(os.getenv("TTS_TIMEOUT_SEC", "20")),
        )
        return Response(content=wav, media_type="audio/wav")
    except asyncio.TimeoutError:
        return JSONResponse({"error": "TTS timeout"}, status_code=503)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=503)

# ルート側ヘルス
@app.get("/health")
def root_health():
    return {"ok": True}

# ルータ登録
app.include_router(gen)
app.include_router(tts)
