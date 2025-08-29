import os, io, wave, numpy as np
from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel
from TTS.api import TTS
import tempfile
from typing import List, Optional
from pathlib import Path
import soundfile as sf
from functools import lru_cache
import hashlib
from contextlib import asynccontextmanager
from fastapi import FastAPI
import torch,warnings
from typing import Any

warnings.filterwarnings("ignore",
    message="pkg_resources is deprecated as an API",
    category=UserWarning, module="jieba._compat")

import asyncio
from asyncio import Semaphore
SEM = Semaphore(1)  # GPUは基本1、CPUなら2〜3にして様子見

_MERGED_WAV_CACHE: dict[tuple[str, int], str] = {}

# プロジェクトルート: .../beatgpt/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
USE_GPU = torch.cuda.is_available()

# ↓ 新: gpu引数は使わず、あとで .to(device)
tts = TTS(model_name=MODEL, progress_bar=False)
device = "cuda" if torch.cuda.is_available() else "cpu"
tts.to(device)
print(f"[boot] XTTS device={device}")

if USE_GPU:
    try:
        tts.model.to("cuda")
        torch.backends.cudnn.benchmark = True
        print("[boot] XTTS on CUDA (half precision)")
    except Exception as e:
        print("[boot] half precision skip:", repr(e))
else:
    # CPU時はスレッドを絞って安定化（Core数に合わせて調整）
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "6")
    os.environ.setdefault("MKL_NUM_THREADS", "6")
    print("[boot] XTTS on CPU")
    

def abs_path(p: str) -> str:
    """相対ならプロジェクトルート基準で絶対化"""
    pp = Path(p)
    return str(pp if pp.is_absolute() else (PROJECT_ROOT / pp).resolve())

app = FastAPI(title="Local Neural TTS (XTTS v2)")

MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
tts = TTS(model_name=MODEL)

# サンプルボイスのマップ
SPEAKER_MAP = {
    "yukari": [
        abs_path("assets/voices/normalized/yukari_1_wave-39-10.wav"),
        abs_path("assets/voices/normalized/yukari_2_009-誰が聞いてあげるのかな？-JmI.wav"),
        abs_path("assets/voices/normalized/yukari_3_030-どこかに、降りれるところ-sDM.wav"),
    ],
    "maki": [
        abs_path("assets/voices/normalized/processed/maki/maki_2_026-今だってボーカロイドのみ-tFm_proc.wav"),
        abs_path("assets/voices/normalized/processed/maki/maki_3_028-ミクさんとかリンちゃんは-ZUh_proc.wav"),
        abs_path("assets/voices/normalized/processed/maki/maki_1_6500_proc.wav"),
        abs_path("assets/voices/normalized/processed/maki/maki_4_114-でも今バフかかってなかっ-fmF_proc.wav"),
    ],
    "ia": [
        abs_path("assets/voices/normalized/IA_1_9000.wav"),
        abs_path("assets/voices/normalized/IA_2_018-歌のお仕事わあんまり好き-Llk.wav"),
        abs_path("assets/voices/normalized/IA_3_021-ゆかりんと一緒にいると、-Ywj.wav"),
        abs_path("assets/voices/normalized/IA_4_028-みんなが傷ついてほしくな-lnI.wav"),
    ],
    "one": [
        abs_path("assets/voices/normalized/ONE_1_10510_2.wav"),
        abs_path("assets/voices/normalized/ONE_2_016-ゆかりちゃんの歌、聞いて-kta.wav"),
        abs_path("assets/voices/normalized/ONE_3_043-真面目な上にしんぱいしょ-WMM.wav"),
    ],
}

def _speaker_map_sanity_check():
    print(f"[boot] PROJECT_ROOT = {PROJECT_ROOT}")
    for spk, lst in SPEAKER_MAP.items():
        for p in lst:
            print(f"[boot] {spk}: {p}  ->  {'OK' if os.path.exists(p) else 'MISSING'}")
_speaker_map_sanity_check()

def get_or_build_merged_wav(paths: list[str], speaker: str, target_sr: int = 24000) -> str:
    key = (speaker, target_sr)
    if key in _MERGED_WAV_CACHE and os.path.exists(_MERGED_WAV_CACHE[key]):
        return _MERGED_WAV_CACHE[key]
    merged = concat_wavs_to_temp(paths, target_sr=target_sr, max_sec=3.5)  # ★
    _MERGED_WAV_CACHE[key] = merged
    return merged

class SynthReq(BaseModel):
    text: str
    speaker: str | None = None
    speaker_wavs: Optional[List[str]] = None
    speed: float = 1.0
    language: str = "ja"

def to_wav_bytes(samples: np.ndarray, sr: int) -> bytes:
    samples = np.clip(samples, -1.0, 1.0).astype(np.float32)
    pcm16 = (samples * 32767).astype(np.int16).tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
        wf.writeframes(pcm16)
    return buf.getvalue()

def read_wav_mono(path: str) -> tuple[np.ndarray, int]:
    x, sr = sf.read(path, dtype='float32', always_2d=True)
    if x.size == 0:
        return np.zeros(0, dtype=np.float32), sr
    x = x.mean(axis=1).astype(np.float32, copy=False)
    return x, sr

def resample_linear(x: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    if sr_from == sr_to:
        return x.astype(np.float32, copy=False)
    ratio = sr_to / sr_from
    n_to = int(round(len(x) * ratio))
    if len(x) == 0 or n_to <= 0:
        return np.zeros(0, dtype=np.float32)
    xp = np.linspace(0.0, 1.0, num=len(x), endpoint=False, dtype=np.float32)
    fp = x.astype(np.float32, copy=False)
    x_new = np.linspace(0.0, 1.0, num=n_to, endpoint=False, dtype=np.float32)
    y = np.interp(x_new, xp, fp).astype(np.float32)
    return y

def concat_wavs_to_temp(paths: list[str], target_sr: int = 24000, max_sec: float = 3.5) -> str:
    chunks = []
    for p in paths:
        try:
            x, sr = read_wav_mono(p)
            x = resample_linear(x, sr, target_sr)
            if x.size:
                chunks.append(x)
        except Exception as e:
            print(f"[concat] skip {p}: {repr(e)}")
    if not chunks:
        raise ValueError("No valid speaker_wavs after filtering")
    y = np.concatenate(chunks)
    # ★ コンディショニングを軽くするため先頭 max_sec 秒だけ使用
    max_len = int(target_sr * max_sec)
    if y.size > max_len:
        y = y[:max_len]
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 0:
        y = (y / peak) * 0.9
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(to_wav_bytes(y.astype(np.float32, copy=False), target_sr))
        return f.name
    

# 起動時にモデルをウォームアップ

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        # 1人分選ぶ（yukari優先→無ければ最初の人）
        spk = "yukari" if "yukari" in SPEAKER_MAP else next(iter(SPEAKER_MAP.keys()))
        paths = [p for p in SPEAKER_MAP[spk] if os.path.exists(p)]
        merged = get_or_build_merged_wav(paths, speaker=spk, target_sr=24000)
        _ = tts.tts(text="あ", language="ja", speed=1.0, speaker_wav=merged)
        print(f"[warmup] OK with speaker_wav ({spk})")
    except Exception as e:
        print("[warmup] skip:", repr(e))
    yield

app = FastAPI(title="Local Neural TTS (XTTS v2)", lifespan=lifespan)

def _warmup():
    try:
        # 1人分選ぶ（yukari優先→無ければ最初の人）
        spk = "yukari" if "yukari" in SPEAKER_MAP else next(iter(SPEAKER_MAP.keys()))
        paths = [p for p in SPEAKER_MAP[spk] if os.path.exists(p)]
        merged = get_or_build_merged_wav(paths, speaker=spk, target_sr=24000)
        _ = tts.tts(text="あ", language="ja", speed=1.0, speaker_wav=merged)
        print("[warmup] TTS model warmed")
    except Exception as e:
        print("[warmup] skip:", repr(e))

#app.post("/synth")


@app.get("/ping")
def ping():
    return {"ok": True}

def _cache_key(text: str, lang: str, speed: float, speaker_id: str, speaker_wav: str | None) -> str:
    base = f"{text}|{lang}|{speed}|{speaker_id}|{speaker_wav or ''}"
    return hashlib.sha1(base.encode("utf-8")).hexdigest()

@lru_cache(maxsize=512)
def _cached_tts(key: str, text: str, lang: str, speed: float, speaker_id: str, speaker_wav: str | None) -> bytes:
    audio = tts.tts(text=text, language=lang, speed=float(speed),
                    **({"speaker_wav": speaker_wav} if speaker_wav else {"speaker": speaker_id}))
    return to_wav_bytes(np.asarray(audio), 24000)


# synth 関数を async def に変更してラップ
@app.post("/synth")
async def synth(req: SynthReq):
    async with SEM:
        return _synth_impl(req)

# synth本体は別関数に分離
def _synth_impl(req: SynthReq):

    if req.speaker_wavs:
        wavs = req.speaker_wavs
    else:
        if req.speaker and req.speaker.lower() in SPEAKER_MAP:
            wavs = SPEAKER_MAP[req.speaker.lower()]
        else:
            print("err0" , req.speaker)
            wavs = None

    kwargs = {}
    if wavs:
        if isinstance(wavs, list):
            paths = [abs_path(p) for p in wavs]
            paths = [p for p in paths if os.path.exists(p)]
            if not paths:
                from fastapi import HTTPException
                print("[synth] no valid speaker_wavs:", wavs)  # デバッグ出力
                raise HTTPException(status_code=400, detail="No valid speaker_wavs after filtering")
            merged = get_or_build_merged_wav(paths, speaker=(req.speaker or "anon"), target_sr=24000)
            kwargs["speaker_wav"] = merged
        elif isinstance(wavs, str):
            path = abs_path(wavs)
            if not os.path.exists(path):
                print("err1" , req)
                from fastapi import HTTPException
                raise HTTPException(status_code=400, detail=f"speaker_wav not found: {path}")
            kwargs["speaker_wav"] = path
    else:
        print("err2" , req)
        # ここでフォールバックはしない。speaker_wav を用意できなければ 400 で返す
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail=f"No speaker_wavs for speaker='{req.speaker}'. Provide speaker_wavs or configure SPEAKER_MAP.")

    speaker_id = (req.speaker or "random")
    speaker_wav = kwargs.get("speaker_wav")
    with torch.inference_mode():
        print(f"[synth] text_len={len(req.text)} spk={req.speaker} kwargs_keys={list(kwargs.keys())}")
        audio = tts.tts(
            text=req.text,
            language=req.language,
            speed=float(req.speed),
            split_sentences=False,  # ★ 追加：文分割を止める
            **({"speaker_wav": speaker_wav} if speaker_wav else {"speaker": speaker_id})
        )
    wav = to_wav_bytes(np.asarray(audio), 24000)
    return Response(content=wav, media_type="audio/wav")

def _dig(obj: Any, path: str):
    cur = obj
    for p in path.split("."):
        cur = getattr(cur, p, None)
        if cur is None:
            return None
    return cur

@app.get("/debug/speakers")
def debug_speakers():
    # ライブラリやモデルによって存在したりしなかったりする属性を総当りチェック
    candidates = {
        "tts.speakers": getattr(tts, "speakers", None),
        "tts.speaker_ids": getattr(tts, "speaker_ids", None),
        "tts.available_speakers": getattr(tts, "available_speakers", None),
        "model.speaker_manager.speaker_ids": _dig(tts, "model.speaker_manager.speaker_ids"),
        "model.speakers": _dig(tts, "model.speakers"),
    }
    out = {}
    for k, v in candidates.items():
        if v is None:
            out[k] = None
        elif isinstance(v, (list, tuple, set)):
            out[k] = list(v)
        else:
            try:
                out[k] = list(v)
            except Exception:
                out[k] = str(type(v))
    return {
        "device": str(getattr(tts, "device", "unknown")),
        "speaker_param_expected": any(bool(v) for v in out.values()),
        "candidates": out,
        "note": "XTTS v2 は通常 speaker_wav で指定します。ここが全部空でも正常です。",
    }
