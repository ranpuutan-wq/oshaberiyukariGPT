# app/tts/cevio_com.py
# CeVIO COM（ITalkerV40）を使って WAV を生成（IA/ONE）
import os, tempfile, pythoncom
import comtypes.client

# 型ライブラリをロード（GUID は公式ドキュメントのlibid参照）
# libid: D3AEA482-B527-4818-8CEA-810AFFCB24B6 （CeVIO CS Talk COM）
tlb = comtypes.client.GetModule(('libid', '{7E3B8901-0A65-44A0-9A9A-5F9F822D0716}', 1, 0))

def _com(f):
    def wrap(*a, **kw):
        pythoncom.CoInitialize()
        try:
            return f(*a, **kw)
        finally:
            pythoncom.CoUninitialize()
    return wrap

@_com
def cevio_talk_wav_bytes(
    text: str,
    cast: str,               # "IA" / "ONE" など
    volume: int = 100,       # 0..100
    speed: int = 50,         # 0..100（標準50付近）
    tone: int = 50,          # 0..100
    tone_scale: int = 50,    # 0..100 （抑揚）
    alpha: int = 50          # 0..100 （声質）
) -> bytes:
    if not text:
        raise ValueError("empty text")

    # ServiceControl / Talker を生成
    svc = comtypes.client.CreateObject(tlb.ServiceControlV40, interface=tlb.IServiceControlV40)
    # CeVIO 起動（接続）
    svc.StartHost(False)  # 起動待ちは IsHostStarted で確認する流儀（公式）:contentReference[oaicite:4]{index=4}

    talker = comtypes.client.CreateObject(tlb.TalkerV40, interface=tlb.ITalkerV40)
    talker.Cast = cast               # 例: "IA", "ONE"（AvailableCastsで列挙可）:contentReference[oaicite:5]{index=5}
    talker.Volume = volume           # 0..100
    talker.Speed = speed             # 0..100
    talker.Tone = tone               # 0..100
    talker.ToneScale = tone_scale    # 0..100
    talker.Alpha = alpha             # 0..100

    # 必要なら感情パラメータ（Components）も設定可能（Cast 依存）:contentReference[oaicite:6]{index=6}
    # ex) comp = talker.Components.ByName("元気"); comp.Value = 80

    # WAVファイルに書き出して bytes で返す（公式が提供しているのはファイル出力API）:contentReference[oaicite:7]{index=7}
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    ok = talker.OutputWaveToFile(text, path)  # 48kHz/16bit/mono 固定
    if not ok:
        raise RuntimeError("CeVIO OutputWaveToFile failed")

    try:
        with open(path, "rb") as f:
            data = f.read()
        if not data or len(data) < 64:
            raise RuntimeError("CeVIO: empty/too small wav")
        return data
    finally:
        try: os.remove(path)
        except: pass

@_com
def cevio_available_casts() -> list[str]:
    svc = comtypes.client.CreateObject(tlb.ServiceControlV40, interface=tlb.IServiceControlV40)
    svc.StartHost(False)
    talker = comtypes.client.CreateObject(tlb.TalkerV40, interface=tlb.ITalkerV40)
    arr = talker.AvailableCasts  # IStringArray
    return [arr.At(i) for i in range(arr.Length)]

