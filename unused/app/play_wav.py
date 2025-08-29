import requests
import simpleaudio as sa

def play_wav_bytes(wav_bytes: bytes) -> None:
    # WAV全体を一気に再生（ブロッキング）
    obj = sa.WaveObject.from_wave_file_bytes(wav_bytes)  # simpleaudio>=1.0.4
    # ↑ 古い版だと from_wave_file_bytes が無いので、その場合は下の fallback を使う
    play = obj.play()
    play.wait_done()

def play_wav_bytes_fallback(wav_bytes: bytes) -> None:
    # simpleaudio 旧版向け（ファイルに落としてから再生）
    import tempfile, os
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(wav_bytes)
        path = f.name
    wave_obj = sa.WaveObject.from_wave_file(path)
    play = wave_obj.play()
    play.wait_done()
    os.remove(path)

if __name__ == "__main__":
    # ▼ どちらかのTTSエンドポイントに合わせて選択 ▼
    # 1) あなたのローカルTTS（XTTS）サーバ: http://127.0.0.1:8020/synth
    url = "http://127.0.0.1:8020/synth"
    payload = {"text":"やっほー！アプリから直接しゃべるテストだよ","speaker":"yukari","speed":1.05}

    # 2) もし TTS Manager を作ってるなら: http://127.0.0.1:8787/speak
    # url = "http://127.0.0.1:8787/speak"
    # payload = {"character_id":"orig_a","text":"TTSマネージャ経由で再生テスト"}

    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()
    wav_bytes = r.content

    try:
        play_wav_bytes(wav_bytes)
    except AttributeError:
        play_wav_bytes_fallback(wav_bytes)

    print("done")
