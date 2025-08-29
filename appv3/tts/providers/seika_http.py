# app/tts/providers/seika_http.py
import os
import urllib.parse
import requests
from dataclasses import dataclass

# AssistSeika 側の設定と合わせる（待ち受けは localhost 推奨。127.0.0.1 だと Invalid Hostname の実装がある）
SEIKA_BASE = os.getenv("SEIKA_BASE", "http://localhost:7180")
SEIKA_USER = os.getenv("SEIKA_USER", "SeikaServerUser")
SEIKA_PASS = os.getenv("SEIKA_PASS", "SeikaServerPassword")
AUTH = (SEIKA_USER, SEIKA_PASS)

@dataclass
class SeikaSelection:
    cid: int
    # 将来拡張用のパラメータ（必要に応じて）
    speed: float = 1.0
    pitch: float = 0.0

def _build_save2_url(base: str, cid: int, text: str, sr: int, speed: float, pitch: float) -> str:
    # text を URLエンコード
    txt = urllib.parse.quote(text, safe="")
    # 実装差に備えてクエリでオプション付加（認識されない環境でも無害）
    return f"{base}/SAVE2/{cid}/{txt}?sr={sr}&speed={speed}&pitch={pitch}&format=wav"

class SeikaHTTP:
    def __init__(self, base: str | None = None):
        self.base = (base or SEIKA_BASE).rstrip("/")

    def synth_wav(self, sel: SeikaSelection, text: str, *, speed=1.0, pitch=0.0, samplerate=48000) -> bytes:
        if not text:
            return b""
        url = _build_save2_url(self.base, sel.cid, text, samplerate, speed, pitch)

        # Host ヘッダを明示（Invalid Hostname 対策）
        host = self.base.split("://", 1)[1]
        headers = {"Host": host}

        r = requests.get(url, auth=AUTH, headers=headers, timeout=120)
        ctype = r.headers.get("Content-Type", "")
        if r.status_code != 200 or not r.content or ("audio" not in ctype and "octet-stream" not in ctype):
            # 失敗時はテキストを見たい（HTML/JSONエラーの頭だけ）
            raise RuntimeError(f"Seika SAVE2 failed: status={r.status_code} type={ctype} body_head={r.text[:200]!r}")

        # 十分なサイズ（1KB超）であることを軽く確認
        if len(r.content) < 1024:
            raise RuntimeError(f"Seika SAVE2 returned too small payload: {len(r.content)} bytes")
        return r.content
