# -*- coding: utf-8 -*-
# app/tts/manager.py
from __future__ import annotations

import os
import asyncio
import logging
import concurrent.futures
from typing import Optional, Dict, Any
import re
from typing import Optional, Dict, Any, Tuple

from app.tts.providers.openai_tts import OpenAITTSProvider
from app.tts.providers.seika_http import SeikaHTTP, SeikaSelection
from app.tts.cevio_sapi import synth_cevio_bytes  # メモリ直接取得版（SpMemoryStream）
from app.tts.cevio_com_dyn import cevio_talk_wav_bytes
from app.tts.cevio_com_dyn import cevio_ai_talk_wav_bytes
from app.tts.aivoice_api import AIVoiceClient, AIVoiceError

log = logging.getLogger(__name__)

EMOTION_CANON_SAMPLE = [
    "neutral",  # 既存
    "joy",      # 既存
    "smile",    # 既存
    "laugh",    # 軽い爆笑/愉快
    "tease",    # からかい/ドヤ
    "surprise", # ちょい驚き
    "confuse",  # 困惑/考え中
    "annoyed",  # ちょいイラ
    "shy",      # 照れ
    "smug",     # ドヤ顔
    "think",    # ふむ/考える
    "sigh",     # はぁ/ため息
]


# --- 話者定義（ここだけを編集すれば経路やIDを変えられる） ---
VOICE_DEF: Dict[str, Dict[str, Any]] = {
#    "yukari": {
#        "voice_software": "seika",
#        "seika_cid": 2003,
#        "cevio_voice": ,
#        "openai_voice": None,
#    },
    "yukari": {
        "voice_software": "aivoice",
        "seika_cid": None, "cevio_voice": "結月 ゆかり", "openai_voice": None,
        "override": {
        #minVol=0.0; maxVol=5.0; aveVol=1.0;
        #minSpd=0.5; maxSpd=4.0; aveSpd=1.0;
        #minPit=0.5; maxPit=2.0; avePit=1.0;
        #minPitchRange=0.0; maxPitchRange=2.0; avePitchRange=1.0;
        "emotion": {
            "neutral":   {"speed": 1.2, "pitch": 1.1, "intonation": 1.3}, # 通常
            "joy":       {"speed": 1.4, "pitch": 1.3, "intonation": 1.3}, # 嬉しい
            "smile":     {"speed": 1.2, "pitch": 1.3, "intonation": 1.3}, # 笑顔
            "laugh":     {"speed": 1.5, "pitch": 1.1, "intonation": 1.3}, # 爆笑
            "tease":     {"speed": 1.2, "pitch": 1.1, "intonation": 1.6}, # からかい
            "surprise":  {"speed": 1.2, "pitch": 1.4, "intonation": 1.6}, # 驚き
            "confuse":   {"speed": 1.4, "pitch": 1.4, "intonation": 1.3}, # 困惑
            "annoyed":   {"speed": 1.0, "pitch": 0.8, "intonation": 1.3}, # イラ 
            "shy":       {"speed": 1.1, "pitch": 1.1, "intonation": 1.5}, # 照れ
            "smug":      {"speed": 1.2, "pitch": 1.1, "intonation": 1.3}, # ドヤ
            "think":     {"speed": 1.0, "pitch": 1.2, "intonation": 1.1}, # 考える
            "sigh":      {"speed": 1.2, "pitch": 0.9, "intonation": 0.8}, # ため息  
        },
        "style": [
            {"when": r"[!！]{2,}", "intonation": 1.4},
            {"when": r"…|……",    "intonation": 0.8},
        ],
        },
    },
    "maki": {
        "voice_software": "seika",
        "seika_cid": 2102, "cevio_voice": None, "openai_voice": None,
        "override": {
        #VoiceRoid2
        #minVol=0.0; maxVol=2.0; aveVol=1.0;
        #minSpd=0.5; maxSpd=4.0; aveSpd=1.0;
        #minPit=0.5; maxPit=2.0; avePit=1.0;
        #minPitchRange=0.0; maxPitchRange=2.0; avePitchRange=1.0;
        "emotion": {
            "neutral":   {"speed": 1.2, "pitch": 1.1, "intonation": 1.3}, # 通常
            "joy":       {"speed": 1.4, "pitch": 1.3, "intonation": 1.3}, # 嬉しい
            "smile":     {"speed": 1.2, "pitch": 1.3, "intonation": 1.3}, # 笑顔
            "laugh":     {"speed": 1.5, "pitch": 1.1, "intonation": 1.3}, # 爆笑
            "tease":     {"speed": 1.2, "pitch": 1.1, "intonation": 1.6}, # からかい
            "surprise":  {"speed": 1.2, "pitch": 1.4, "intonation": 1.6}, # 驚き
            "confuse":   {"speed": 1.4, "pitch": 1.4, "intonation": 1.3}, # 困惑
            "annoyed":   {"speed": 1.0, "pitch": 0.8, "intonation": 1.3}, # イラ 
            "shy":       {"speed": 1.1, "pitch": 1.1, "intonation": 1.5}, # 照れ
            "smug":      {"speed": 1.2, "pitch": 1.1, "intonation": 1.3}, # ドヤ
            "think":     {"speed": 1.0, "pitch": 1.2, "intonation": 1.1}, # 考える
            "sigh":      {"speed": 1.2, "pitch": 0.9, "intonation": 0.8}, # ため息  
        },
        "style": [
            {"when": r"[!！]{2,}", "intonation": 1.4},
            {"when": r"…|……",    "intonation": 0.8},
        ],
        },
    },
#    "maki": {
#        "voice_software": "cevioai",
#        "seika_cid": None,
#        "cevio_voice": "弦巻マキ (日)",  # CeVIO AI のキャスト名
#        "openai_voice": None,
#        "override": {
#        "emotion": {
#            "joy":   {"speed": 1.10, "pitch": +0.05, "intonation": 70},
#            "think": {"speed": 0.95, "pitch": -0.03, "intonation": 40},
#        },
#        "style": [
#            {"when": r"[!！]{2,}", "intonation": 75},
#            {"when": r"…|……",    "intonation": 45},
#        ],
#        },
#    },
    "ia": {
        "voice_software": "cevio",
        "seika_cid": None,
        "cevio_voice": "IA",
        "openai_voice": None,  # フォールバックに使うなら "coral"/"shimmer" 等を入れる
        "override": {
        #CevioCS
        #minVol=-8.0; maxVol=8.0; aveVol=0.0;
        #minSpd= 0.2; maxSpd=5.0; aveSpd=1.0;
        #minPit=-6.0; maxPit=6.0; avePit=0.0;
        #minPitchRange=0.0; maxPitchRange=2.0; avePitchRange=1.0;
        "emotion": {
            "neutral":   {"speed": 1.6, "pitch": 1.3, "intonation": 1.3}, # 通常
            "joy":       {"speed": 2.2, "pitch": 1.9, "intonation": 1.3}, # 嬉しい
            "smile":     {"speed": 1.6, "pitch": 1.9, "intonation": 1.3}, # 笑顔
            "laugh":     {"speed": 2.5, "pitch": 1.3, "intonation": 1.3}, # 爆笑
            "tease":     {"speed": 1.6, "pitch": 1.3, "intonation": 1.6}, # からかい
            "surprise":  {"speed": 1.6, "pitch": 2.2, "intonation": 1.6}, # 驚き
            "confuse":   {"speed": 3.2, "pitch": 2.2, "intonation": 1.3}, # 困惑
            "annoyed":   {"speed": 1.0, "pitch": 0.6, "intonation": 1.3}, # イラ 
            "shy":       {"speed": 1.3, "pitch": 1.3, "intonation": 1.5}, # 照れ
            "smug":      {"speed": 1.6, "pitch": 1.3, "intonation": 1.3}, # ドヤ
            "think":     {"speed": 1.0, "pitch": 1.6, "intonation": 1.1}, # 考える
            "sigh":      {"speed": 1.6, "pitch": 0.8, "intonation": 0.8}, # ため息  
        },
        "style": [
            {"when": r"[!！]{2,}", "intonation": 1.4},
            {"when": r"…|……",    "intonation": 0.8},
        ],
        },
    },
    "one": {
        "voice_software": "cevio",
        "seika_cid": None,
        "cevio_voice": "ONE",
        "openai_voice": None,  # フォールバックに使うなら設定
        "override": {
        #CevioCS
        #minVol=-8.0; maxVol=8.0; aveVol=0.0;
        #minSpd= 0.2; maxSpd=5.0; aveSpd=1.0;
        #minPit=-6.0; maxPit=6.0; avePit=0.0;
        #minPitchRange=0.0; maxPitchRange=2.0; avePitchRange=1.0;
        "emotion": {
            "neutral":   {"speed": 1.6, "pitch": 1.3, "intonation": 1.3}, # 通常
            "joy":       {"speed": 2.2, "pitch": 1.9, "intonation": 1.3}, # 嬉しい
            "smile":     {"speed": 1.6, "pitch": 1.9, "intonation": 1.3}, # 笑顔
            "laugh":     {"speed": 2.5, "pitch": 1.3, "intonation": 1.3}, # 爆笑
            "tease":     {"speed": 1.6, "pitch": 1.3, "intonation": 1.6}, # からかい
            "surprise":  {"speed": 1.6, "pitch": 2.2, "intonation": 1.6}, # 驚き
            "confuse":   {"speed": 3.2, "pitch": 2.2, "intonation": 1.3}, # 困惑
            "annoyed":   {"speed": 1.0, "pitch": 0.6, "intonation": 1.3}, # イラ 
            "shy":       {"speed": 1.3, "pitch": 1.3, "intonation": 1.5}, # 照れ
            "smug":      {"speed": 1.6, "pitch": 1.3, "intonation": 1.3}, # ドヤ
            "think":     {"speed": 1.0, "pitch": 1.6, "intonation": 1.1}, # 考える
            "sigh":      {"speed": 1.6, "pitch": 0.8, "intonation": 0.8}, # ため息  
        },
        "style": [
            {"when": r"[!！]{2,}", "intonation": 1.4},
            {"when": r"…|……",    "intonation": 0.8},
        ],
        },
    },
    # 例）OpenAI直： "rin": {"voice_software":"openai","openai_voice":"shimmer",...}
}

# 経路ごとの並列上限（セマフォで制御）
PARALLEL_LIMITS = {
    "seika": 1,
    "cevio": 1,
    "cevioai": 1,   # ← 追加
    "openai": 4,
    "aivoice": 1,  # A.I.VOICE は重いので1
}

# Seika のデフォルトサンプリングレート
SEIKA_SAMPLERATE = 48000


class TTSManager:
    """
    VOICE_DEF に基づいてルーティングし、各プロバイダへ委譲する TTS マネージャ。
    - 同期 API: synth_for(...)
    - 非同期 API: synth_for_async(...)  ※内部で thread pool + 経路別セマフォ
    """

    def __init__(self) -> None:
        # プロバイダ初期化
        self.seika = SeikaHTTP()
        self.openai = OpenAITTSProvider()
        # ルート初期化時に AIVOICE クライアントの遅延生成
        self._aivoice = None
        
        # スレッドプール（ブロッキング実装を非同期化する）
        self._pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=sum(PARALLEL_LIMITS.values()) + 2
        )

        # 経路別セマフォ
        self._sema: Dict[str, asyncio.Semaphore] = {
            route: asyncio.Semaphore(PARALLEL_LIMITS[route]) for route in PARALLEL_LIMITS
        }

        # ルーティング確認ログ
        log.info("[TTSManager] initialized routes=%s", list(self._sema.keys()))

    # ---------- 内部ユーティリティ ----------
    @staticmethod
    def _norm_speaker(s: str) -> str:
        return (s or "").strip().lower()

    def _conf_for(self, speaker: str) -> Dict[str, Any]:
        spk = self._norm_speaker(speaker)
        conf = VOICE_DEF.get(spk)
        if not conf:
            raise RuntimeError(f"Unknown speaker='{speaker}'. Define it in VOICE_DEF.")
        return conf

    def _route_for(self, speaker: str) -> str:
        conf = self._conf_for(speaker)
        route = conf.get("voice_software")
        if route not in PARALLEL_LIMITS:
            raise RuntimeError(f"Unsupported route '{route}' for speaker='{speaker}'.")
        return route

    def _ensure_aivoice(self):
        if self._aivoice is None:
            # 既定はインストール検知に任せる。特定名で縛りたいなら prefer_host="A.I.VOICE"
            self._aivoice = AIVoiceClient()
        return self._aivoice
    

    # ---------- 同期版（既存呼び出し互換） ----------
    def synth_for(
        self,
        speaker: str,
        text: str,
        *,
        style: Optional[str] = None,   # 現状未使用（拡張用）
        speed: float = 1.0,
        pitch: float = 0.0,            # Seika のみ対応
        emotion: Optional[str] = None, # 予約（CeVIO RemoteService/C#ブリッジで活用可）
    ) -> bytes:
        """
        同期で各プロバイダに合成を依頼して bytes を返す。
        エラー時は例外を投げる（呼び出し側でキャッチ or 非同期ラッパでログ化）。
        """

        print ("[DEBUG] TTSManager.synth_for: speaker=%s text=%s style=%s speed=%.2f pitch=%.2f emotion=%s" % (
            speaker, text, style, speed, pitch, emotion
            )
        )

        conf = self._conf_for(speaker)
        # 追加：override を適用
        speed, pitch, intonation, eff = self._apply_overrides(speaker, text, emotion, speed, pitch)

        route = conf["voice_software"]
        spk = self._norm_speaker(speaker)
        text = (text or "").strip()

        print("[DEBUG] TTSManager.synth_for: route=%s speaker=%s speed=%.2f pitch=%.2f intonation=%s text=%s" % (
            route, spk, speed, pitch, intonation, text
            )
        )

        if not text:
            raise ValueError("Empty text for TTS")

        if route == "seika":
            cid = conf.get("seika_cid")
            if cid is None:
                raise RuntimeError(f"speaker='{speaker}' has no seika_cid")
            sel = SeikaSelection(cid=int(cid))
            log.debug("[tts] route=SEIKA speaker=%s cid=%s speed=%.2f pitch=%.2f", spk, cid, speed, pitch)
            # SeikaHTTP はバイト列の WAV を返す想定
            cid = eff.get("seika_cid", conf.get("seika_cid"))
            return self.seika.synth_wav({"cid": cid}, text, speed=speed, pitch=pitch)


        if route == "cevio":
            cast = eff.get("cevio_voice", conf.get("cevio_voice"))
            tone_scale = int(round(intonation if intonation is not None else 50))
            tone_scale = max(0, min(100, tone_scale))            
            return cevio_talk_wav_bytes(
                text, cast=cast, volume=100, speed=int(round(50 + (speed - 1.0) * 50)),
                tone=50, tone_scale=tone_scale, alpha=50
            )

        # ★ CeVIO AI 追加ブロック
        if route == "cevioai":
            cast = eff.get("cevio_voice", conf.get("cevio_voice"))
            tone_scale = int(round(intonation if intonation is not None else 50))
            tone_scale = max(0, min(100, tone_scale))
            return cevio_ai_talk_wav_bytes(
                text, cast=cast, volume=100, speed=int(round(50 + (speed - 1.0) * 50)),
                tone=50, tone_scale=tone_scale, alpha=50
            )
        
        if route == "openai":
            voice = conf.get("openai_voice") or "shimmer"  # デフォルトを好みに
            log.debug("[tts] route=OPENAI speaker=%s voice=%s speed=%.2f", spk, voice, speed)
            voice = eff.get("openai_voice", conf.get("openai_voice"))
            return self.openai.synthesize(text, voice=voice, speed=speed)

        # 既存の synth_for 内の switch に追加:
        elif route == "aivoice":
            cli = self._ensure_aivoice()
            print("[DEBUG] TTSManager.route aivoice cli= %s speed = %.2f pitch=%.2f intonation=%s" % (cli, speed, pitch, intonation))
            return cli.synth_bytes(
                text,
                voice_name=conf.get("cevio_voice") or conf.get("aivoice_voice"),
                speed=speed, pitch=pitch,
                intonation=intonation,  # ← new!
            )

        raise RuntimeError(f"No valid route for speaker='{speaker}'")

    # ---------- 非同期ラッパ（FastAPI などから await で呼べる） ----------
    async def synth_for_async(
        self,
        speaker: str,
        text: str,
        *,
        style: Optional[str] = None,
        speed: float = 1.0,
        pitch: float = 0.0,
        emotion: Optional[str] = None,
    ) -> bytes:
        """
        - 経路別セマフォで同時実行数を制限
        - ブロッキング処理はスレッドプールで実行
        """
        route = self._route_for(speaker)
        async with self._sema[route]:
            loop = asyncio.get_running_loop()
            try:
                return await loop.run_in_executor(
                    self._pool,
                    lambda: self.synth_for(
                        speaker, text,
                        style=style, speed=speed, pitch=pitch, emotion=emotion
                    ),
                )
            except Exception as e:
                log.error("[tts] synth failed route=%s speaker=%s err=%s", route, speaker, e)
                raise

    # ---------- デバッグ/運用補助 ----------
    def describe_voice(self, speaker: str) -> Dict[str, Any]:
        """この話者の現在のルーティング設定を返す（ログ/テスト用）"""
        conf = self._conf_for(speaker).copy()
        conf["route"] = conf.get("voice_software")
        return conf

    def set_route(self, speaker: str, route: str, **kwargs) -> None:
        """
        実行中にルートを切り替えたいとき（例：CeVIO未インストール環境で一時的にOpenAIへ）
        kwargsで openai_voice / seika_cid / cevio_voice を同時更新可
        """
        spk = self._norm_speaker(speaker)
        if route not in PARALLEL_LIMITS:
            raise ValueError(f"Unknown route '{route}'")
        if spk not in VOICE_DEF:
            raise ValueError(f"Unknown speaker '{speaker}'")
        VOICE_DEF[spk]["voice_software"] = route
        for k, v in kwargs.items():
            if k in ("openai_voice", "seika_cid", "cevio_voice"):
                VOICE_DEF[spk][k] = v
        log.info("[tts] set_route %s -> %s (%s)", spk, route, kwargs)

    def _apply_overrides(
        self, spk: str, text: str, emotion: Optional[str],
        speed: float, pitch: float
    ) -> Tuple[float, float, Optional[float], Dict[str, Any]]:
        """
        VOICE_DEF.override に基づいて speed/pitch/intonation/voice系を上書き。
        返り値: (speed, pitch, intonation, eff_voices)
        intonation: 0-100 の共通値 or None
        """
        conf = self._conf_for(spk)
        ov = conf.get("override") or {}
        eff = {
            "openai_voice": conf.get("openai_voice"),
            "seika_cid":    conf.get("seika_cid"),
            "cevio_voice":  conf.get("cevio_voice"),
        }
        s, p = speed, pitch
        itn: Optional[float] = None  # 0-100 共通

        # 1) emotion 連動
        if emotion:
            rule = (ov.get("emotion") or {}).get(emotion)
            if rule:
                if "speed" in rule: s = float(rule["speed"])
                if "pitch" in rule: p = float(rule["pitch"])
                if "intonation" in rule: itn = float(rule["intonation"])
                if "openai_voice" in rule: eff["openai_voice"] = rule["openai_voice"]
                if "seika_cid" in rule:    eff["seika_cid"]    = rule["seika_cid"]
                if "cevio_voice" in rule:  eff["cevio_voice"]  = rule["cevio_voice"]

        # 2) style風（text 正規表現）連動：最初にヒットした1件だけ適用
        for pat in (ov.get("style") or []):
            w = pat.get("when")
            if not w: 
                continue
            try:
                if re.search(w, text):
                    if "speed" in pat: s = float(pat["speed"])
                    if "pitch" in pat: p = float(pat["pitch"])
                    if "intonation" in pat: itn = float(pat["intonation"])
                    if "openai_voice" in pat: eff["openai_voice"] = pat["openai_voice"]
                    if "seika_cid" in pat:    eff["seika_cid"]    = pat["seika_cid"]
                    if "cevio_voice" in pat:  eff["cevio_voice"]  = pat["cevio_voice"]
                    break
            except re.error:
                pass

        return s, p, itn, eff
    
