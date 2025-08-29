#OPENAI モデル名
#順位	音声名	性別傾向	主な特徴（参考）
#1	Onyx	男性的	深みがあり権威のある声
#2	Nova	男性的	明るくエネルギッシュな男性の声
#3	Sage	やや男性的/中性的	落ち着いた知的な声
#4	Fable	やや男性的/中性的	魅力的な語り手の声
#5	Echo	中性的	温かく自然な声
#6	Ash	中性的	親しみやすく会話的な声
#7	Verse	中性的	表現力豊かでニュートラルな声
#8	Alloy	中性的	性別認識が曖昧で多目的な声
#9	Ballad	やや女性的/中性的	上品でメロディアスな声
#10	Coral	やや女性的/中性的	明るく前向きな声
#11	Shimmer	女性的	クリアで表現力豊かな女性の声

# app/tts/manager.py
# ルーティング:
# - yukari -> A.I.VOICE (SAPI5)
# - maki   -> VOICEROID2 (SAPI5)
# - ia     -> OpenAI TTS (voice="coral")
# - one    -> OpenAI TTS (voice="shimmer")
#
# 依存:
# - Windows で SAPI5 を使うため pywin32 が必要:  pip install pywin32
# - OpenAI API キー: 環境変数 OPENAI_API_KEY
#
# 既知の注意:
# - SAPI5 はピッチ指定が弱いので pitch は未対応（rate/volume のみ反映）
# - OpenAI 側の voice は "coral" / "shimmer" など

from __future__ import annotations
import os
import io
import tempfile
import requests
from typing import Optional

# --- SAPI5 (Windows) 用 ---
HAS_SAPI = False
try:
    import win32com.client  # type: ignore
    from win32com.client import constants  # type: ignore
    HAS_SAPI = True
except Exception:
    HAS_SAPI = False


# ======================
# Utility
# ======================
def _float_speed_to_sapi_rate(speed: float) -> int:
    """
    speed=1.0 を基準に SAPI の Rate (-10..+10) にマップ
    1.0 → 0,  0.9 → -2,  1.1 → +2 くらいの緩い変換
    """
    speed = max(0.5, min(1.8, float(speed or 1.0)))
    if speed >= 1.0:
        return min(10, int((speed - 1.0) * 20))  # 1.0→0, 1.5→+10
    else:
        return max(-10, int((speed - 1.0) * 20))  # 0.5→-10


def _read_file_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


# ======================
# SAPI5 Backend (A.I.VOICE / VOICEROID2)
# ======================
class SAPIBackend:
    """
    SAPI5 経由で wav を生成。voice_hint に部分一致する音声を選ぶ。
    例:
      - A.I.VOICE 結月ゆかり → "yukari" / "結月" / "A.I.VOICE"
      - VOICEROID2 弦巻マキ  → "maki" / "弦巻" / "VOICEROID"
    """
    def __init__(self):
        if not HAS_SAPI:
            raise RuntimeError("pywin32 が見つかりません（Windows + 'pip install pywin32' が必要）")

    def _pick_voice_token(self, hint_substr: str | None):
        sp = win32com.client.Dispatch("SAPI.SpVoice")
        voices = sp.GetVoices()
        if hint_substr:
            key = hint_substr.lower()
            for i in range(voices.Count):
                v = voices.Item(i)
                try:
                    desc = v.GetDescription()
                except Exception:
                    desc = ""
                if key in desc.lower():
                    return v
        # fallback: 最初の音声
        return voices.Item(0) if voices.Count > 0 else None

    def synth_to_wav(self, text: str, voice_hint: Optional[str], speed: float = 1.0, volume: int = 100) -> bytes:
        sp = win32com.client.Dispatch("SAPI.SpVoice")
        tok = self._pick_voice_token(voice_hint or "")
        if tok is not None:
            sp.Voice = tok
        sp.Rate = _float_speed_to_sapi_rate(speed)  # -10..+10
        sp.Volume = max(0, min(100, int(volume)))

        stream = win32com.client.Dispatch("SAPI.SpFileStream")
        # 形式を指定（48k/16bit/Mono）。指定不可の場合は既定にフォールバック
        try:
            stream.Format.Type = getattr(constants, "SAFT48kHz16BitMono")
        except Exception:
            pass

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            out_path = f.name
        # SSFMCreateForWrite=3
        stream.Open(out_path, 3, False)
        sp.AudioOutputStream = stream
        sp.Speak(text)
        stream.Close()
        wav = _read_file_bytes(out_path)
        try:
            os.remove(out_path)
        except Exception:
            pass
        return wav


# ======================
# OpenAI TTS Backend (REST)
# ======================
class OpenAIBackend:
    """
    OpenAI Audio/Speech REST を直接叩く（SDK差異回避）
    POST https://api.openai.com/v1/audio/speech
    body: {model: "gpt-4o-mini-tts", voice: "shimmer", input: "...", format: "wav"}
    """
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini-tts"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY が未設定です。環境変数に設定してください。")
        self.model = model
        self.url = "https://api.openai.com/v1/audio/speech"

    def synth_wav(self, text: str, voice: str = "shimmer") -> bytes:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "voice": voice,
            "input": text,
            "format": "wav",
        }
        r = requests.post(self.url, headers=headers, json=payload, timeout=120)
        if r.status_code != 200:
            raise RuntimeError(f"OpenAI TTS error {r.status_code}: {r.text[:300]}")
        return r.content


# ======================
# Local XTTS Backend (既存 /synth)
# ======================
class XTTSBackend:
    def __init__(self, base_url: str = "http://127.0.0.1:8020"):
        self.url = base_url.rstrip("/") + "/synth"

    def synth(self, text: str, speaker: str, speed: float = 1.0) -> bytes:
        payload = {"text": text, "language": "ja", "speaker": speaker, "speed": speed}
        r = requests.post(self.url, json=payload, timeout=180)
        if r.status_code != 200:
            raise RuntimeError(f"XTTS error {r.status_code}: {r.text[:300]}")
        return r.content


# ======================
# TTS Manager
# ======================
class TTSManager:
    """
    キャラ→エンジンのルーティングをここで集中管理。
    - ゆかり  : SAPI(A.I.VOICE)   voice_hint="yukari" など
    - マキ    : SAPI(VOICEROID2)  voice_hint="maki"
    - IA     : OpenAI voice="coral"
    - ONE    : OpenAI voice="shimmer"
    """
    def __init__(self):
        # Backends
        self._openai = OpenAIBackend(model=os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts"))
        self._xtts: Optional[XTTSBackend] = None
        if os.getenv("XTTS_URL"):
            self._xtts = XTTSBackend(os.getenv("XTTS_URL"))

        self._sapi: Optional[SAPIBackend] = None
        if HAS_SAPI:
            try:
                self._sapi = SAPIBackend()
            except Exception as e:
                print("[TTSManager] SAPI init failed:", repr(e))

        # ルーティングテーブル
        # voice_hint: SAPI 用の部分一致キーワード（環境に合わせて調整してOK）
        self.routing = {
            "yukari": {"engine": "sapi", "voice_hint": os.getenv("AIVOICE_VOICE_HINT", "yukari")},
            "maki":   {"engine": "sapi", "voice_hint": os.getenv("VOICEROID_VOICE_HINT", "maki")},
            "ia":     {"engine": "openai", "voice": os.getenv("IA_OPENAI_VOICE", "coral")},
            "one":    {"engine": "openai", "voice": os.getenv("ONE_OPENAI_VOICE", "shimmer")},
        }

    # メインのエントリ
    def synth_for(
        self,
        character_id: str,
        text: str,
        style: Optional[str] = None,
        speed: float = 1.0,
        pitch: float = 0.0,  # SAPIは未サポート
        emotion: Optional[str] = None,
    ) -> bytes:
        cid = (character_id or "").strip().lower()
        cfg = self.routing.get(cid, None)

        if not cfg:
            # 未定義キャラ → OpenAI (shimmer) にフォールバック
            return self._openai.synth_wav(text, voice="shimmer")

        engine = cfg.get("engine")

        if engine == "openai":
            voice = cfg.get("voice", "shimmer")
            return self._openai.synth_wav(text, voice=voice)

        if engine == "sapi":
            if not self._sapi:
                raise RuntimeError("SAPI backend が利用できません（Windows + pywin32 が必要）")
            voice_hint = cfg.get("voice_hint", "")
            # volume は 0..100。速度のみ反映
            return self._sapi.synth_to_wav(text, voice_hint=voice_hint, speed=speed, volume=100)

        if engine == "xtts":
            if not self._xtts:
                raise RuntimeError("XTTS backend が未設定です。環境変数 XTTS_URL を設定してください。")
            spk = cfg.get("speaker", cid)
            return self._xtts.synth(text, speaker=spk, speed=speed)

        # 未知 → OpenAI
        return self._openai.synth_wav(text, voice="shimmer")

