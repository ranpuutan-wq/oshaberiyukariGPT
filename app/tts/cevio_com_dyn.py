# app/tts/cevio_com_dyn.py
import pythoncom, time
import os, tempfile, pythoncom
import win32com.client as win32

def cevio_talk_wav_bytes(text: str, cast: str, *, volume=100, speed=50, tone=50, tone_scale=60, alpha=50) -> bytes:
    if not text:
        raise ValueError("empty text")

    pythoncom.CoInitialize()
    try:
        # ここがポイント：TypeLib不要。ProgID で生成
        # 環境により V50/V41/V40 など異なるので、手当て付きで探す
        for ver in ("V50","V41","V40"):
            try:
                svc = win32.Dispatch(f"CeVIO.Talk.RemoteService.ServiceControl{ver}")
                talker = win32.Dispatch(f"CeVIO.Talk.RemoteService.Talker{ver}")
                break
            except Exception:
                svc = talker = None
        if not talker:
            raise RuntimeError("CeVIO COM が見つかりません（CeVIO CS のインストール/bit数を確認）")
        
        svc.StartHost(False)  # 起動／接続

        # キャスト設定（AvailableCasts で確認可）
        talker.Cast = cast
        talker.Volume = int(volume)
        talker.Speed = int(speed)          # 0..100（標準=50）
        talker.Tone = int(tone)
        talker.ToneScale = int(tone_scale)
        talker.Alpha = int(alpha)

        print ("[DEBUG] CeVIO CS: Cast=%s Speed=%d Tone=%d ToneScale=%d Alpha=%d Text=%s "% (
            talker.Cast, talker.Speed, talker.Tone, talker.ToneScale, talker.Alpha, text)
        )

        fd, path = tempfile.mkstemp(suffix=".wav"); os.close(fd)
        ok = talker.OutputWaveToFile(text, path)  # 48kHz/16bit/mono
        if not ok:
            raise RuntimeError("CeVIO OutputWaveToFile failed (Cast名が違う可能性)")

        with open(path, "rb") as f:
            data = f.read()
        if not data or len(data) < 64:
            raise RuntimeError("CeVIO: empty/too small wav")
        return data
    finally:
        try: os.remove(path)
        except: pass
        pythoncom.CoUninitialize()


def _connect_talker():
    # バージョンを順に試す（環境で V50/V41/V40 など違う）
    for ver in ("V50", "V41", "V40"):
        try:
            svc = win32.Dispatch(f"CeVIO.Talk.RemoteService.ServiceControl{ver}")
            talker = win32.Dispatch(f"CeVIO.Talk.RemoteService.Talker{ver}")
            return svc, talker, ver
        except Exception:
            pass
    raise RuntimeError("CeVIO COM(ServiceControl/Talker)が見つからないッピ")

def cevio_available_casts() -> list[str]:
    pythoncom.CoInitialize()
    try:
        svc, talker, ver = _connect_talker()
        # CeVIO 本体を起動/接続（起動待ちも挟む）
        svc.StartHost(False)
        for _ in range(50):  # 最大5秒待つ
            try:
                arr = talker.AvailableCasts
                if getattr(arr, "Length", 0) > 0:
                    return [arr.At(i) for i in range(arr.Length)]
            except Exception:
                pass
            time.sleep(0.1)
        return []  # 見つからなければ空
    finally:
        pythoncom.CoUninitialize()

def _connect_talker2():
    # 環境ごとに V50/V41/V40/無印の可能性がある
    for ver in ("V50", "V41", "V40", ""):
        try:
            svc2 = win32.Dispatch(f"CeVIO.Talk.RemoteService2.ServiceControl2{ver}")
            talker2 = win32.Dispatch(f"CeVIO.Talk.RemoteService2.Talker2{ver}")
            return svc2, talker2, ver
        except Exception:
            pass
    raise RuntimeError("CeVIO AI COM(ServiceControl2/Talker2)が見つからないッピ")

def cevio_ai_talk_wav_bytes(text: str, cast: str, *, volume=100, speed=50, tone=50, tone_scale=60, alpha=50) -> bytes:
    """
    CeVIO AI (RemoteService2/Talker2) で合成して WAV bytes を返す
    """
    if not text:
        raise ValueError("empty text")

    pythoncom.CoInitialize()
    try:
        svc2, talker2, ver = _connect_talker2()
        svc2.StartHost(False)

        talker2.Cast = cast
        talker2.Volume = int(volume)
        talker2.Speed = int(speed)
        talker2.Tone = int(tone)
        talker2.ToneScale = int(tone_scale)
        talker2.Alpha = int(alpha)

        print ("[DEBUG] CeVIO AI: Cast=%s Speed=%d Tone=%d ToneScale=%d Alpha=%d Text=%s "% (
            talker2.Cast, talker2.Speed, talker2.Tone, talker2.ToneScale, talker2.Alpha, text)
        )

        fd, path = tempfile.mkstemp(suffix=".wav"); os.close(fd)
        ok = talker2.OutputWaveToFile(text, path)
        if not ok:
            raise RuntimeError("CeVIO AI OutputWaveToFile failed (Cast名が違う可能性)")

        with open(path, "rb") as f:
            data = f.read()
        if not data or len(data) < 64:
            raise RuntimeError("CeVIO AI: empty/too small wav")
        return data
    finally:
        try: os.remove(path)
        except: pass
        pythoncom.CoUninitialize()

def cevio_ai_available_casts() -> list[str]:
    pythoncom.CoInitialize()
    try:
        svc2, talker2, ver = _connect_talker2()
        svc2.StartHost(False)
        for _ in range(50):
            try:
                arr = talker2.AvailableCasts
                if getattr(arr, "Length", 0) > 0:
                    return [arr.At(i) for i in range(arr.Length)]
            except Exception:
                pass
            time.sleep(0.1)
        return []
    finally:
        pythoncom.CoUninitialize()
        