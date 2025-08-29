# app/tts/aivoice_api.py
import os
import tempfile
import time
import logging

# まずは COM で攻める。ProgID は「AI.Talk.Editor.Api.TtsControl」でOKな構成
# もし失敗したら pythonnet 経由に自動フォールバック（下に実装）
try:
    import win32com.client as win32
    _com_ok = hasattr(win32, "Dispatch")
    if not _com_ok:
        win32 = None
except Exception:
    win32 = None

log = logging.getLogger(__name__)

class AIVoiceError(RuntimeError):
    pass

class AIVoiceClient:
    """
    A.I.VOICE Editor API ラッパー（単純化）
    - Host 検出 → 接続
    - 音声一覧取得
    - 合成してWAV/bytesで返す
    """
    def __init__(self, prefer_host: str | None = None):
        self.prefer_host = prefer_host  # 例: "A.I.VOICE"
        self._ctl = None
        self._connected = False

    # --- 初期化 & 接続 ---
    def connect(self):
        # 1) TtsControl 生成
        self._ctl = self._create_ttscontrol()

        # 2) 利用可能ホストを取得
        hosts = list(self._ctl.GetAvailableHostNames())
        if not hosts:
            raise AIVoiceError("A.I.VOICE: 利用可能なホストが見つかりません。")

        if self.prefer_host:
            host = next((h for h in hosts if self.prefer_host.lower() in h.lower()), hosts[0])
        else:
            # 一般的には "A.I.VOICE" が含まれる名前が返る
            host = next((h for h in hosts if "voice" in h.lower()), hosts[0])

        # 3) Initialize & 起動 & 接続
        self._ctl.Initialize(host)                 # 初期化（ホスト名指定）  :contentReference[oaicite:3]{index=3}
        if self._ctl.Status == 0:                  # HostStatus.NotRunning っぽいとき
            self._ctl.StartHost()                  # ホスト起動            :contentReference[oaicite:4]{index=4}
        self._ctl.Connect()                        # 接続                  :contentReference[oaicite:5]{index=5}
        self._connected = True
        log.info("[AIVOICE] connected host=%s ver=%s", host, getattr(self._ctl, "Version", "?"))

    def _create_ttscontrol(self):
        # ✅ まず COM（推奨）
        if win32 is not None:
            try:
                return win32.Dispatch("AI.Talk.Editor.Api.TtsControl")
            except Exception as e:
                log.warning("[AIVOICE] COM作成失敗: %r -> pythonnet fallback", e)

        # ✅ ここから pythonnet フォールバック（DLL直読み）
        dll_path = os.getenv("AIVOICE_API_DLL")
        if not dll_path or not os.path.exists(dll_path):
            raise AIVoiceError(
                "A.I.VOICE API: COM生成不可 & AIVOICE_API_DLL 未設定/不在。"
                " 環境変数 AIVOICE_API_DLL に AI.Talk.Editor.Api.dll のフルパスを設定してください。"
            )
        import clr  # pythonnet
        clr.AddReference(dll_path)
        from AI.Talk.Editor.Api import TtsControl  # type: ignore
        return TtsControl()

    # --- キャスト列挙 ---
    def list_voices(self) -> list[str]:
        self._ensure()
        try:
            # VoiceNames / CurrentVoiceName が一般的（ドキュメント準拠）
            # 実プロパティ名は API 版によって微妙に違うことがあるため両対応
            names = getattr(self._ctl, "VoiceNames", None)
            if names is None:
                # 代替: ボイスプリセット名だけでも拾う
                names = getattr(self._ctl, "VoicePresetNames", None)
            if names is None:
                return []
            return list(names)
        except Exception as e:
            log.warning("[AIVOICE] list_voices error: %r", e)
            return []

    # --- 合成（WAV bytes） ---
    def synth_bytes(self, text: str, voice_name: str | None = None,
                    speed: float | None = None, pitch: float | None = None,
                    volume: float | None = None, intonation: float | None = None) -> bytes:
        """
        - voice_name を設定（あれば）
        - マスターコントロール(JSON文字列)を簡易で反映（必要なら）
        - SaveAudioToFile(path) でファイルへ保存（同期）→ bytes で返却
        """
        print ("[DEBUG] AIVoiceClient.synth_bytes: voice_name=%s speed=%.2f pitch=%.2f volume=%.2f intonation=%.2f text=%s" % (
            voice_name, speed or 1.0, pitch or 1.0, volume or 1.0, intonation or 1.0, text
            )
        )
        self._ensure()
        if voice_name:
            try:
                # 例: _ctl.CurrentVoiceName など
                if hasattr(self._ctl, "CurrentVoiceName"):
                    self._ctl.CurrentVoiceName = voice_name
                elif hasattr(self._ctl, "CurrentVoicePresetName"):
                    self._ctl.CurrentVoicePresetName = voice_name
            except Exception as e:
                log.warning("[AIVOICE] set voice failed: %s -> %r", voice_name, e)

        # 必要ならマスターコントロールを JSON で設定できる（速度/ピッチなど）
        # 公式は JSON での授受を推奨（Volume/Speed/Pitch/PitchRange…）:contentReference[oaicite:6]{index=6}
        # Volume=0~5、Speed=0.5～4.0、Pitch=0.5～2.0、PitchRange=0.0～2.0 の範囲内に入るように 入力値をそのままmcに設定する
        minVol=0.0; maxVol=5.0; aveVol=1.0;
        minSpd=0.5; maxSpd=4.0; aveSpd=1.0;
        minPit=0.5; maxPit=2.0; avePit=1.0;
        minPitchRange=0.0; maxPitchRange=2.0; avePitchRange=1.0;
        
        if any(v is not None for v in (speed, pitch, volume)):
            mc = {
#                "Volume": 1.0 if volume is None else float(volume),
#                "Speed":  1.0 if speed  is None else float(speed),
#                "Pitch":  1.0 if pitch  is None else float(pitch),
                "Volume": aveVol if volume is None else max(minVol, min(maxVol, volume)),
                "Speed": aveSpd if speed  is None else max(minSpd, min(maxSpd, float(speed))),
                "Pitch": avePit if pitch  is None else max(minPit, min(maxPit, float(pitch))),
                "PitchRange": avePitchRange if intonation is None else max(minPitchRange, min(maxPitchRange, float(intonation))),
            }

        print("[DEBUG] AIVoiceClient.synth_bytes2: Voice=%s Speed=%.2f Pitch=%.2f Volume=%.2f Intonation=%.2f Text=%s" % (
            voice_name, mc.get("Speed", 1.0), mc.get("Pitch", 1.0), mc.get("Volume", 1.0), mc.get("PitchRange", 1.0), text
            )
        )
        try:
            import json
            self._ctl.MasterControl = json.dumps(mc)
        except Exception as e:
            log.warning("[AIVOICE] set MasterControl failed: %r", e)

        # SaveAudioToFile は同期。完了までブロックする仕様:contentReference[oaicite:7]{index=7}
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            out_path = f.name
        try:
            # テキストを設定して保存
            self._ctl.Text = text
            self._ctl.SaveAudioToFile(out_path)     # 同期保存            :contentReference[oaicite:8]{index=8}

            # ファイルを bytes で返す
            with open(out_path, "rb") as r:
                data = r.read()
            return data
        finally:
            try:
                os.remove(out_path)
            except Exception:
                pass

    def _ensure(self):
        if not self._connected or self._ctl is None:
            self.connect()

    def close(self):
        if self._ctl is not None:
            try:
                self._ctl.Disconnect()
            except Exception:
                pass
        self._connected = False
