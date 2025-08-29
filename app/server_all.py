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

#debug
import time, logging
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
    
    # デバッグ
    t0 = time.perf_counter()
    route = mgr.describe_voice(req.speaker).get("route")  # どのエンジンに行くか
    log.info("[/tts/speak] start speaker=%s route=%s len(text)=%d",
            req.speaker, route, len(req.text))
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
        dt = time.perf_counter() - t0
        log.info("[/tts/speak] done  speaker=%s route=%s bytes=%s dt=%.2fs",
                 req.speaker, route, (len(wav) if wav else 0), dt)
        return Response(content=wav, media_type="audio/wav")
    
    except asyncio.TimeoutError:
        dt = time.perf_counter() - t0
        log.warning("[/tts/speak] TIMEOUT speaker=%s route=%s dt=%.2fs",
                    req.speaker, route, dt)
        return JSONResponse({"error": "TTS timeout"}, status_code=503)
    
    except Exception as e:
        dt = time.perf_counter() - t0
        log.exception("[/tts/speak] ERROR   speaker=%s route=%s dt=%.2fs err=%s",
                      req.speaker, route, dt, e)
        return JSONResponse({"error": str(e)}, status_code=503)

# ルート側ヘルス
@app.get("/health")
def root_health():
    return {"ok": True}

# ルータ登録
app.include_router(gen)
app.include_router(tts)


#司会追加
class ProposeReq(BaseModel):
    topic: str
    history: list[dict] = []     # [{"speaker":"...","text":"..."}... 直近数件でOK]
    speakers: list[str] = []     # ["yukari","maki","ia","one"]

@gen.post("/propose")
def propose(req: ProposeReq):
    # ここは app.llm_openai に “1発言だけ返す” 軽プロンプト関数を用意して呼ぶ想定
    # 返すJSONは [{"speaker","text","emotion","priority","can_overlap"} ...]
    # 実装は手元の generate_turns を縮約 or 専用プロンプト関数を作成
    proposals = []  # ← 実装する
    return {"proposals": proposals}
