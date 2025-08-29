# app/tts/cevio_sapi.py
import pythoncom
import win32com.client as win32
import comtypes.client

# SAPI 定数
SpeechLib = comtypes.client.GetModule(r"C:\Windows\System32\Speech\Common\sapi.dll")

def _with_com(f):
    # 呼び出しスレッドで STA 初期化/解放
    def wrap(*args, **kwargs):
        pythoncom.CoInitialize()
        try:
            return f(*args, **kwargs)
        finally:
            pythoncom.CoUninitialize()
    return wrap

def _pick_cevio_voice_id(hint: str) -> str | None:
    sp = win32.Dispatch("SAPI.SpVoice")
    hint_low = (hint or "").lower()
    # 優先: CeVIO かつ hint（"IA","ONE"など）を含む
    for v in sp.GetVoices():
        desc = v.GetDescription()
        if "cevio" in desc.lower() and hint_low in desc.lower():
            return v.Id
    # 次点: CeVIO のどれか
    for v in sp.GetVoices():
        if "cevio" in v.GetDescription().lower():
            return v.Id
    return None

@_with_com
def synth_cevio_bytes(text: str, *, voice_hint: str, rate: int = 0, volume: int = 100) -> bytes:
    sp = win32.Dispatch("SAPI.SpVoice")
    vid = _pick_cevio_voice_id(voice_hint)
    if not vid:
        raise RuntimeError(f"CeVIOのSAPI音源が見つかりません（hint={voice_hint}）。CeVIO CSが起動/インストール済みか確認してね。")

    sp.Voice = sp.GetVoices(f"ID={vid}").Item(0)
    sp.Rate = max(-10, min(10, rate))
    sp.Volume = max(0, min(100, volume))

    # メモリストリームに 48k/16bit/mono で吐かせる
    mem = win32.Dispatch("SAPI.SpMemoryStream")
    fmt = win32.Dispatch("SAPI.SpAudioFormat")
    fmt.Type = SpeechLib.SpeechAudioFormatType.SAFT48kHz16BitMono
    mem.Format = fmt

    sp.AudioOutputStream = mem
    sp.Speak(text)

    data = bytes(mem.GetData())  # ← ファイルを経由しない
    if not data or len(data) < 64:
        raise RuntimeError("CeVIO: 合成できたがデータが小さすぎる/空です")
    return data
