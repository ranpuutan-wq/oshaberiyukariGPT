# app/tts/providers/sapi.py
# SAPI5 経由で A.I.VOICE / VOICEROID2 を利用（Windows専用）
import os, tempfile
import win32com.client  # pip install pywin32
from typing import Optional, List

class SapiVoiceProvider:
    def __init__(self):
        # SAPI.SPVoice / SAPI.SpFileStream
        self.spvoice = win32com.client.Dispatch("SAPI.SPVoice")
        self.file_stream = win32com.client.Dispatch("SAPI.SpFileStream")

    def list_voices(self) -> List[str]:
        tokens = self.spvoice.GetVoices()
        names = []
        for i in range(tokens.Count):
            token = tokens.Item(i)
            names.append(token.GetDescription())
        return names

    def _pick_token_by_hint(self, hint_substring: str):
        tokens = self.spvoice.GetVoices()
        hint = (hint_substring or "").lower()
        # まず日本語っぽいのを優先
        best = None
        for i in range(tokens.Count):
            t = tokens.Item(i)
            name = t.GetDescription()
            lower = name.lower()
            if hint and hint in lower:
                return t
            # ヒントが無い場合は "ja" っぽいものを候補に
            if not hint and ("ja" in lower or "japanese" in lower or "日本" in lower):
                best = best or t
        # 見つからなければ先頭
        return best or (tokens.Item(0) if tokens.Count > 0 else None)

    def synthesize(self, text: str, *, voice_hint: str = "", speed: float = 1.0) -> bytes:
        token = self._pick_token_by_hint(voice_hint)
        if token is None:
            raise RuntimeError("No SAPI voice is available on this system.")
        # voice 設定
        self.spvoice.Voice = token

        # rate: SAPIは -10..+10 の整数。speed を 0.5〜1.5 を想定して rate にマップ
        # 例: 1.0 -> 0, 1.2 -> +2, 0.8 -> -2
        rate = max(-10, min(10, int(round((speed - 1.0) * 10))))
        self.spvoice.Rate = rate

        # WAVへ書き出し
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            path = f.name
        from comtypes.gen import SpeechLib  # 自動生成されない場合は一度実行して生成される
        self.file_stream.Open(path, SpeechLib.SSFMCreateForWrite)
        self.spvoice.AudioOutputStream = self.file_stream
        # Flags: 0 = Default
        self.spvoice.Speak(text, 0)
        self.spvoice.WaitUntilDone(0)  # ブロッキング
        self.file_stream.Close()

        with open(path, "rb") as rf:
            data = rf.read()
        try:
            os.remove(path)
        except Exception:
            pass
        return data
