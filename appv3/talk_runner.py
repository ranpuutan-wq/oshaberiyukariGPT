import requests
import time
import simpleaudio as sa
import tempfile, os
import re
import random
from openai import OpenAI
from seika_http_client import SeikaClient

from difflib import SequenceMatcher
from dataclasses import dataclass, field
from typing import Optional

from audio_queue import AudioQueue
import threading, re


GEN_URL = os.getenv("GEN_URL", "http://127.0.0.1:8787/gen/generate_4")
TTS_URL = os.getenv("TTS_URL", "http://127.0.0.1:8787/tts/speak")

#QUEUES = {spk: AudioQueue() for spk in ["yukari","maki","ia","one"]}

# 一人称・呼び名（あなたの定義に合わせて）
SELF_PRON = {"yukari": "私", "maki": "私", "ia": "IA", "one": "私"}
CALL_MAP = {
    "yukari": {"maki": "マキさん", "ia": "IAちゃん", "one": "ONEさん"},
    "maki":   {"yukari": "ゆかりちゃん", "ia": "IAちゃん", "one": "ONEちゃん"},
    "ia":     {"yukari": "ゆかりん", "maki": "マキ", "one": "ONE"},
    "one":    {"yukari": "ゆかり", "maki": "マキ", "ia": "IA"},
}

SEIKA = SeikaClient(host="127.0.0.1", port=7180,
                    user="SeikaServerUser", password="SeikaServerPassword")

# AssistSeikaの /AVATOR2 結果から控えてくれたCID
SEIKA_CID = {
    "yukari": 2101,  # 例: 結月ゆかり_解説_VOICEROID2_2101
    "maki":   2102,  # 例: 弦巻マキ_実況_VOICEROID2_2102
    # IA/ONE は OPENAI or AIVOICE 予定なので未設定でもOK
}

DOCROOT = r"C:\Docroot"  # AssistSeikaの「ドキュメントルートフォルダ」に合わせる

OPENAI_TTS_VOICE = "verse"  # 例: "verse" や "alloy" など
OPENAI_TTS_MODEL = "gpt-4o-mini-tts"  # OpenAI TTS用モデル名（必要に応じて変更）

# 自分の名前の表記ゆれ（文頭主語→一人称に直す対象は“自分だけ”）
SELF_ALIASES = {
    "yukari": [r"(?:結月ゆかり|ゆかり|Yukari)"],
    "maki":   [r"(?:弦巻マキ|マキ|Maki)"],
    "ia":     [r"(?:IA|ＩＡ|Ia|ia)"],
    "one":    [r"(?:ONE|ＯＮＥ|One|one)"],
}
# 形式ばった言い回し→砕け口語に置換（共通）
FORMAL2CASUAL = [
    (r"一体何の理由で", "なんで"),
    (r"理由は", ""),                      # 先頭の「理由は」を落として自然化
    (r"つまり、?", ""),                   # 「つまり」を省いて口語化
    (r"比較してみて", "比べるなら"),
    (r"決まりましたね", "決まりじゃん"),
    (r"楽しみですね", "楽しみだね"),
]

# 語尾崩し候補（ランダムで差し替え）
ENDINGS_CASUAL = ["だね", "じゃん", "かな", "かも", "でしょ", "よね", "いこ", "しよ"]
QUESTION_TAILS = ["どう？", "じゃない？", "あり？", "で良くない？", "にする？"]

# JK口語フィルタ：翻訳調 → 砕け日本語
JK_REPLACEMENTS = [
    (r"つまり、?", ""),                   # 「つまり」消す
    (r"効率的に", "うまく"),               # 硬い表現を柔らかく
    (r"必要がある", "じゃん"),              # 決まり文句を崩す
    (r"同じ感じよ", "同じだよ〜"),          # 翻訳調を崩す
    (r"方法", "やり方"),                   # method → やり方
    (r"紹介したい", "教えるよ"),            # announce系を砕く
    (r"聞きたいです", "聞きたい！"),        # 丁寧 → 砕け
    (r"短縮する", "短くする"),              # formal → everyday
    (r"次回の話題は『(.+?)』だな", r"次は「\1」にしよっか"),  # 締めのフォーマルを砕く
]

TOPIC_POOL = [
    # 学校/日常
    "体育祭の出し物どうする？",
    "放課後の時間つぶし案",
    "週末どこ行く？（近場）",
    "写真の撮り方コツ",
    "新しい勉強法、続くやつある？",
    # 趣味/創作
    "歌ってみた選曲どうする？",
    "動画の編集ソフト、何使う？",
    "自転車のコース開拓",
    "旅行の持ち物の最小セット",
    "最近ハマったゲームの語り",
    # テック/ライフハック
    "スマホの通知どう整理する？",
    "朝のルーティン短縮術",
    "AIの使いどころ",
    "お金の管理アプリ",
    "集中切れたときの戻し方",
]


_openai_client = None
def _client_once():
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _openai_client

def tts_openai(text: str, voice: str = "alloy", model: str = "gpt-4o-mini-tts") -> bytes:
    client = _client_once()
    resp = client.audio.speech.create(
        model=model,
        voice=voice,
        input=text,
        format="wav",
    )
    return resp.read()

def _tts_openai_via_http(text: str, voice: str = "alloy",
                         model: str = "gpt-4o-mini-tts",
                         response_format: str = "wav") -> bytes:
    url = "https://api.openai.com/v1/audio/speech"
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY','')}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "voice": voice,
        "input": text,
        "response_format": response_format,   # ← 同じく response_format
    }
    r = requests.post(url, headers=headers, json=payload, timeout=120)
    r.raise_for_status()
    return r.content


def _tts_openai_via_sdk(text: str, voice: str = "alloy",
                        model: str = "gpt-4o-mini-tts",
                        response_format: str = "wav") -> bytes:
    client = _client_once()
    resp = client.audio.speech.create(
        model=model,
        voice=voice,
        input=text,
        response_format=response_format,   # ← ここがポイント！（format ではない）
    )
    # SDKは .read() が使える版と .content を持つ版がある
    return getattr(resp, "read", lambda: resp.content)()

@dataclass
class ConvState:
    decided: Optional[bool] = None  # True=合意/方針決定 / None=未決
    # “候補名”はトピック非依存化のため保持しない（将来拡張するなら抽出器を差す）

# “決定っぽい語” と “行動着手っぽい語”（ジャンルを問わず汎用）
DECISION_PAT = r"(決まり|決めよ|決めよう|にする|で行こう|採用|これでいこう|一旦これ)"
ACTION_PAT   = r"(買ってくる|行ってくる|予約する|申し込む|送る|発注する|投稿する|公開する|実装する|開始する|着手する|発進|購入する|決済する)"

def update_state_with_line(state: ConvState, text: str):
    # “決定語”を観測したら decided=True
    if re.search(DECISION_PAT, text):
        state.decided = True

def enforce_consistency(state: ConvState, speaker: str, text: str) -> str:
    t = text
    # 未決のうちは “行動着手語” をやわらかく抑制（下見/相談に弱める）
    if state.decided is None and re.search(ACTION_PAT, t):
        t = re.sub(ACTION_PAT, "一回だけ様子見しよ？", t)
    # 決定後は“迷走”を抑制（『結局どうする？』を提案に変える等）
    if state.decided:
        t = re.sub(r"(どうする\?|どうしよ\?|結局どうする)", "じゃ、進めよ。細かいのはあとで決めよ", t)
    return t

def pick_topic() -> str:
    return random.choice(TOPIC_POOL)

# 直近と同じ語尾の連続を避ける
def _diversify_ending(text: str, last_text: str | None) -> str:
    if not last_text:
        return text
    end = re.findall(r"(だね|ですね|じゃん|かな|かも|です|でしょう)[！!。]?$", text)
    last_end = re.findall(r"(だね|ですね|じゃん|かな|かも|です|でしょう)[！!。]?$", last_text)
    if end and last_end and end[-1] == last_end[-1]:
        t = re.sub(r"(だね|ですね|じゃん|かな|かも|です|でしょう)[！!。]?$", "", text)
        new_end = random.choice(ENDINGS_CASUAL + QUESTION_TAILS)
        return (t + new_end)
    return text

# 非ゆかりの「ですね」を禁止（ゆかりのみ許可）
def _ban_desune_for_non_yukari(speaker: str, text: str) -> str:
    if speaker == "yukari":
        return text
    return re.sub(r"ですね([！!。]?)$", r"だね\1", text)

# 他者の呼び方を統一（例: “IAさん”→“IAちゃん” 等）
def _normalize_calls(speaker: str, text: str) -> str:
    for other, name in CALL_MAP[speaker].items():
        # ゆるく: 相手の英字/カナ/漢字っぽい揺れをまとめて置換
        patt = {
            "yukari": r"(?:結月ゆかり|ゆかりん|ゆかり|Yukari)",
            "maki":   r"(?:弦巻マキ|マキさん|マキ|Maki)",
            "ia":     r"(?:IAちゃん|IAさん|IA)",
            "one":    r"(?:ONEちゃん|ONEさん|ONE|One)",
        }[other]
        text = re.sub(patt, name, text)
    return text

# 自分の一人称を強制（文頭の「私は/わたしは/ワタシは/マキは…」→自分の一人称へ）
def _normalize_self_pron(speaker: str, text: str) -> str:
    me = SELF_PRON[speaker]
    t = text

    # 文頭「自分の名前は…」だけ一人称に変える（他人名は絶対対象外）
    for patt in SELF_ALIASES.get(speaker, []):
        t = re.sub(rf"^\s*(?:{patt})\s*は", me + "は", t)

    # 代表的一人称ゆれ → 自分の一人称へ
    t = re.sub(r"^\s*(私|わたし|ワタシ|あたし)\s*は", me + "は", t)

    return t


def jk_filter(text: str) -> str:
    t = text
    for patt, repl in JK_REPLACEMENTS:
        t = re.sub(patt, repl, t)
    return t

# 翻訳調の定型を砕く
def _formal_to_casual(text: str) -> str:
    for patt, rep in FORMAL2CASUAL:
        text = re.sub(patt, rep, text)
    # 句読点整形（全角。の重複など）
    text = re.sub(r"[。\.]{2,}$", "。", text)
    return text.strip()

def polish_line(speaker: str, text: str, last_text: str | None) -> str:
    """1行を“ルール準拠のJK口語”へ後処理"""
    if not text:
        return text
    t = text.strip()

    # 4) 一人称を統一（自分）
    t = _normalize_self_pron(speaker, t)

    # 3) 呼び名を統一（相手）
    t = _normalize_calls(speaker, t)

    # 1) 固い言い回し→砕く
    t = _formal_to_casual(t)

    # 2) 非ゆかりの「ですね」禁止
    t = _ban_desune_for_non_yukari(speaker, t)

    # 追加: JKっぽく砕く
    t = jk_filter(t)
    
    # 5) 語尾の多様化（直前と同じ語尾を避ける）
    t = _diversify_ending(t, last_text)

    return t

# ループっぽさ検出用パターン
NG_PATTS = [
    r"(?:.+?)がいいね[。！!？\?]*$",
    r"(?:.+?)にしよ[うー]*[。！!？\?]*$",
    r"(?:.+?)がいい[。！!？\?]*$",
]

def is_loop_like(new_text: str, history_texts: list[str], sim_thresh: float = 0.87) -> bool:
    """末尾パターン連打 or 類似度高すぎならループ判定"""
    t = re.sub(r"\s+", "", new_text)
    # 語尾パターンの連打
    for p in NG_PATTS:
        if re.search(p, t):
            recent = "".join(re.sub(r"\s+", "", s) for s in history_texts[-6:])
            if len(re.findall(p, recent)) >= 2:
                return True
    # 高類似（直近と文字列類似）
    for prev in history_texts[-6:]:
        s = SequenceMatcher(a=re.sub(r"\s+","",prev), b=t).ratio()
        if s >= sim_thresh:
            return True
    return False

ALT_ENDINGS = ["どう？", "ってどう思う？", "あり？", "にする案もあるよ。", "もう一個だけ足すと…"]

def light_paraphrase(text: str) -> str:
    """簡易言い換え：語尾の多様化＋質問化で前進させる"""
    t = re.sub(r"(?:だよ|だね|かな|かも|ですよね)[。！!]*$", "", text)
    t = re.sub(r"(?:がいいね|がいい)$", "もあり", t)
    return (t + "、" + random.choice(ALT_ENDINGS)).strip()

# ---- その1: 正規化マップ ----
CANON = {
    "結月ゆかり": "yukari", "ゆかり": "yukari", "yukari": "yukari",
    "弦巻マキ": "maki", "マキ": "maki", "maki": "maki",
    "IA": "ia", "ia": "ia", "イア": "ia",
    "ONE": "one", "one": "one", "オネ": "one",
}
CHAR_ORDER = ["yukari", "maki", "ia", "one"]

PROSODY_RULES = [
    (r"(けど|だけど|だけどさ)(?!、)", r"\1、"),
    (r"(から|ので|なら)(?!、)", r"\1、"),
    (r"([ぁ-んァ-ン一-龥A-Za-z0-9]{5,})(?!、)(?=は|も|が|を)", r"\1、"),
    (r"([!?！？])", r"\1 "),           # 記号後の軽いスペースで間
    (r"([。\.])(?!\s)", r"\1 "),       # ピリオド後にもスペース
    (r"…(?! )", "… "),                # 伸ばしの後に間
]


def normalize_spk(s: str) -> str:
    if not s:
        return "yukari"
    return CANON.get(s.strip(), s.strip().lower())

# ---- TTS呼び出し用 ----
#TTS_URL = "http://127.0.0.1:8020/synth"  # ← /synth に直す！（いまは /tts で404）  :contentReference[oaicite:4]{index=4}

def tts_bytes(text: str, speaker: str, emotion: str, speed: float = 1.02 ) -> bytes | None:
    # /tts/speak 用のボディ（manager 経由）
    payload = {"speaker": speaker, "text": text, "speed": speed, "emotion": emotion}
    r = requests.post(TTS_URL, json=payload, timeout=120)
    if r.status_code != 200:
        print(f"[TTS ERROR] status={r.status_code} body[:200]={r.text[:200]}")
        return None
    # audio/wav が来る前提
    if r.headers.get("content-type","").startswith("audio/"):
        return r.content
    print(f"[TTS ERROR] unexpected content-type={r.headers.get('content-type')}")
    return None

def play_bytes(wav_bytes: bytes):
    if not wav_bytes:
        print("[play_bytes] skip: no data")
        return

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        try:
            f.write(wav_bytes)
            path = f.name
        except Exception as e:
            print(f"[play_bytes] write failed: {e}")
            return

    try:
        wave = sa.WaveObject.from_wave_file(path)
        play = wave.play()
        play.wait_done()
    except Exception as e:
        print(f"[play_bytes] playback failed: {e}")
    finally:
        try:
            os.remove(path)
        except Exception as e:
            print(f"[play_bytes] cleanup failed: {e}")


def soft_pick_speaker(
    model_spk: str | None,
    history_speakers: list[str],
    seen: set[str],
    chars: list[str],
    turns_left: int,
) -> str:
    """強制しない“ベストエフォート4人登場”の話者選定"""
    # モデルの出力が不正ならなし扱い
    if model_spk not in chars:
        model_spk = None

    unseen = [c for c in chars if c not in seen]

    weights: dict[str, float] = {}
    for c in chars:
        w = 1.0

        # まだ出てない人を少しだけ優遇
        if c in unseen:
            w *= 1.8

        # 直前と同じ人は弱める（連投を抑制）
        if history_speakers and c == history_speakers[-1]:
            w *= 0.35

        # 直近2ターンに居たら少し弱める（会話の回りをよくする）
        if len(history_speakers) >= 2 and c in (history_speakers[-2],):
            w *= 0.7

        # モデルが選んだ話者は基本尊重（自然さを優先）
        if model_spk == c:
            w *= 1.3

        # 終盤で未登場が残っていたら、でも“強制”せずブーストだけ強める
        if c in unseen and turns_left <= len(unseen) + 1:
            w *= 1.5

        weights[c] = w

    # 重み付きサンプリング（強制ではない）
    total = sum(weights.values())
    r = random.random() * total
    acc = 0.0
    for c, w in weights.items():
        acc += w
        if r <= acc:
            return c
    return chars[0]

# ---- 会話実行 ----
def run_conversation(topic=None, turns=12):
    if topic is None:
        topic = pick_topic()

    r = requests.post(GEN_URL, json={"topic": topic, "turns": turns}, timeout=60)
    if r.status_code != 200:
        print("[debug] HTTP", r.status_code)
        print("[debug] body[:500] =", r.text[:500])
        return

    data = r.json()
    turns_data = data.get("turns", [])

    print(f"[Topic] {topic}\n")
    print(f"turns_data length: {len(turns_data)}")
    print(f"original talk: {turns_data}")

    state = ConvState() 
    history_speakers = [] # 直近の話者履歴
    history_texts = [] # 直近の発話履歴（テキストのみ）
    
    seen = set()        # これまでに登場した話者
    
    
    for i, t in enumerate(turns_data):
        raw_spk = t.get("speaker", "")
        spk = normalize_spk(raw_spk)

        # …テキスト生成・polish_line など既存処理…
        text = t.get("text", "").strip()
        update_state_with_line(state, text)     # ← 先に状態更新
        text = enforce_consistency(state, spk, text)  # ← その状態で整合性適用
        
        # ループっぽさ検出→軽い言い換え
        if is_loop_like(text, history_texts):
            text = light_paraphrase(text)
            # 仕上げに再度ポリッシュ＆整合性
            text = polish_line(spk, text, history_texts[-1] if history_texts else None)
            update_state_with_line(state, text)
            text = enforce_consistency(state, spk, text)
            if not text:
                text = "うーん、ちょっと考えとくね。"
        if not text:
            text = "うーん、ちょっと考えとくね。"

        text = text.replace("\n", " ").strip()

        if not text:
            text = "…"  # 空白回避 
        if len(text) > 120:
            text = text[:120] + "…" # 長すぎ回避

        # print(f"Raw speaker: '{raw_spk}' -> Normalized: '{spk}'")
        
        
        # 補正は「不正な名前」かつ「未登場キャラが残ってる」場合だけ
        turns_left = len(turns_data) - i
        if spk not in CHAR_ORDER:
            spk = soft_pick_speaker(
                model_spk=None,
                history_speakers=history_speakers,
                seen=seen,
                chars=CHAR_ORDER,
                turns_left=turns_left,
            )

        print(f"[{spk}] {text}")
        seen.add(spk)
        history_speakers.append(spk)
        history_texts.append(text)

        line = text   # 発話用に間を補う

        # TTS再生
        wav = tts_bytes(line, spk, speed=1.08, emotion=t.get("emotion","neutral"))
        if not wav:
            print("[runner] TTS failed; skipping this line")
            continue

        play_bytes(wav)
        time.sleep(0.15)

def main():
    topic = None
    turns = 12
    print(f"[runner] start topic={topic} turns={turns}")
    try:
        run_conversation(topic=topic, turns=turns)
        print("[runner] done.")
    except Exception:
        import traceback
        print("[runner] EXCEPTION:")
        traceback.print_exc()

if __name__ == "__main__":
    main()

