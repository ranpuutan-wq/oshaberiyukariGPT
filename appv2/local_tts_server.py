import os, io, wave, numpy as np
from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel
from TTS.api import TTS
import tempfile
from typing import List, Optional
import numpy as np, wave, io, os, tempfile

from pathlib import Path
import soundfile as sf

# プロジェクトルート: .../beatgpt/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

def abs_path(p: str) -> str:
    """相対ならプロジェクトルート基準で絶対化"""
    pp = Path(p)
    return str(pp if pp.is_absolute() else (PROJECT_ROOT / pp).resolve())

app = FastAPI(title="Local Neural TTS (XTTS v2)")

MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
tts = TTS(model_name=MODEL)

# サンプルボイスのマップ
# 短く 空白がなく 3つくらいまで
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



class SynthReq(BaseModel):
    text: str
    speaker: str | None = None
    speaker_wavs: Optional[List[str]] = None   # ← これを追加
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
    # dtype='float32' で -1..1 に正規化されて返る / always_2d で (N, ch)
    x, sr = sf.read(path, dtype='float32', always_2d=True)
    if x.size == 0:
        return np.zeros(0, dtype=np.float32), sr
    x = x.mean(axis=1).astype(np.float32, copy=False)  # モノラル化
    return x, sr


def resample_linear(x: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    """軽量な線形補間リサンプリング"""
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


def concat_wavs_to_temp(paths: list[str], target_sr: int = 24000) -> str:
    """複数WAVを読み込み→モノラル＆target_srに整列→連結→一時WAVのパスを返す"""
    chunks = []
    for p in paths:
        try:
            x, sr = read_wav_mono(p)
            print ("[concat1]",x,p)

            print(f"[concat] after read: size={x.size} sr={sr}")
            x = np.clip(x, -1.0, 1.0).astype(np.float32, copy=False)
            x = resample_linear(x, sr, target_sr)
            print(f"[concat] after resample: size={x.size}")
            if x.size:
                chunks.append(x)
        except Exception as e:
            print(f"[concat] skip {p}: {repr(e)}")
    if not chunks:
        print ("[concat2]",paths)
        print ("[concat3]",chunks)
        
        raise ValueError("No valid speaker_wavs after filtering")
    
    y = np.concatenate(chunks)
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 0:
        y = (y / peak) * 0.9  # 軽く正規化
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        # 既存の to_wav_bytes を再利用
        f.write(to_wav_bytes(y.astype(np.float32, copy=False), target_sr))
        tmp_path = f.name
    return tmp_path

@app.post("/synth")
def synth(req: SynthReq):
    # --- speaker_wavsが指定されていれば優先 ---
    if req.speaker_wavs:
        wavs = req.speaker_wavs
    else:
        # speakerキーをSPEAKER_MAPから解決
        if req.speaker and req.speaker.lower() in SPEAKER_MAP:
            wavs = SPEAKER_MAP[req.speaker.lower()]
        else:
            wavs = None

    kwargs = {}
    if wavs:
        
        # リストの場合 → 複数をマージ
        if isinstance(wavs, list):

            # 例: speaker_wavs または SPEAKER_MAP から来たパス群を絶対化して存在チェック
            paths = [abs_path(p) for p in wavs]  # ← 絶対化
            paths = [p for p in paths if os.path.exists(p)]
            if not paths:
                print("[synth] no valid speaker_wavs:", wavs)  # デバッグ出力
                raise ValueError("No valid speaker_wavs after filtering")
            print ("[synth] valid speaker_wavs:", len(paths))

            if paths:
                merged = concat_wavs_to_temp(paths, target_sr=24000)
                kwargs["speaker_wav"] = merged
        # 単一パスの場合
        elif isinstance(wavs, str) and os.path.exists(wavs):
            kwargs["speaker_wav"] = wavs
    else:
        kwargs["speaker"] = "english"

    audio = tts.tts(
        text=req.text,
        language=req.language,
        speed=req.speed,
        **kwargs
    )
    wav = to_wav_bytes(np.asarray(audio), 24000)
    return Response(content=wav, media_type="audio/wav")


@app.get("/ping")
def ping():
    return {"ok": True}

