import requests
from audio_queue import AudioQueue

TTS_URL = "http://127.0.0.1:8020/synth"  # or http://127.0.0.1:8787/speak

def synth(text: str) -> bytes:
    r = requests.post(TTS_URL, json={"text": text, "speed": 1.05}, timeout=120)
    r.raise_for_status()
    return r.content

if __name__ == "__main__":
    aq = AudioQueue()
    lines = [
        "よし、順番にしゃべらせてみよう！",
        "二つ目のセリフいきまーす",
        "最後はオチっぽく、ふふっ"
    ]
    for t in lines:
        aq.enqueue(synth(t))

    print("待機中…")
    aq.wait_all()
    print("全部再生完了！")

