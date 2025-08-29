import requests
from audio_queue import AudioQueue

GEN_URL = "http://127.0.0.1:8787/generate_4"
TTS_URL = "http://127.0.0.1:8020/synth"

# 話者ごとに速度や声を微調整したい場合のマップ
SPEAKER_MAP = {
    "yukari": {"speaker": "orig_a", "speed": 1.00},
    "maki":   {"speaker": "orig_b", "speed": 1.05},
    "ia":     {"speaker": "orig_c", "speed": 1.10},
    "one":    {"speaker": "orig_d", "speed": 0.98},
}
DEFAULT = {"speaker": None, "speed": 1.05}

def tts(text: str, spk_opts: dict) -> bytes:
    payload = {
        "text": text,
        "speed": spk_opts.get("speed", 1.05)
    }
    if spk_opts.get("speaker"):
        payload["speaker"] = spk_opts["speaker"]  # XTTSの speaker_wav マップをサーバ側で対応しておく
    r = requests.post(TTS_URL, json=payload, timeout=120)
    r.raise_for_status()
    return r.content

if __name__ == "__main__":
    # 4人会話生成
    r = requests.post(GEN_URL, json={"topic": "今日のお昼なに食べる？", "turns": 12}, timeout=60)
    r.raise_for_status()
    data = r.json()

    aq = AudioQueue()
    for turn in data["turns"]:
        spk = turn["speaker"]
        text = turn["text"]
        opts = SPEAKER_MAP.get(spk, DEFAULT)
        wav = tts(text, opts)
        aq.enqueue(wav)
        print(f"[{spk}] {text}")

    aq.wait_all()
    print("会話再生完了！")
