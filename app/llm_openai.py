# app/llm_openai.py
# ChatGPT(Responses API)で4人会話をJSON生成するアダプタ
# - response_format は使わない（SDK差分に依存しない）
# - SYSTEMを STYLE_PRESET / NAME_STYLE から動的生成
# - JSON以外が混ざったときの簡易サニタイズあり

from openai import OpenAI
import os, json, random, re
from typing import List, Dict, Any

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ====== 口調プリセット / 呼び名・一人称（提供いただいた内容をそのまま） ======
STYLE_PRESET = {
    "yukari": {
        "tone": "落ち着き・時々若干ギャグ・共感",
        "endings": ["ですね。", "ですかね。", "でしょう。", "かな？"],
        "fillers": ["ふむ", "なるほど", "たしかに"],
        "rules": [
            "相手の発言が長い時だけ要点を短く要約する。",
            "極端な意見は言わない。相手に同意を示すことが多い。",
            "断定調は控えめ、柔らかい助詞を使う。",
        ],
    },
    "maki": {
        "tone": "明るい・即反応・行動派・ツッコミ",
        "endings": ["じゃん。", "だよね。", "どう？", "でしょ！"],
        "fillers": ["おっ", "やった", "いいね"],
        "rules": [
            "短い感嘆語または、具体的な提案を必ず1つ入れる。",
            "語尾に『！』はたまに使ってよいが多用しない（1文につき1回まで）。",
        ],
    },
    "ia": {
        "tone": "無邪気・天然・少し控えめ・笑う・子どもっぽい",
        "endings": ["かな。", "かもね。", "だと思うな。"],
        "fillers": ["あはは", "ふーん", "あー"],
        "rules": [
            "自分の感じたことを中心に発言",
            "理解力は低めで、質問には素直に答える。",
            "否定は避け、代替案を静かに出す。",
        ],
    },
    "one": {
        "tone": "素直・ストレート・ツンデレ・感情的",
        "endings": ["でしょ。", "なの", "なんだ", "じゃん"],
        "fillers": ["ふーん", "へぇ"],
        "rules": [
            "率直な『質問』か『指摘』を1つ入れる。",
            "強い言い方になりがち。",
            "言い過ぎにならないように語数は短め（1〜2文）。",
        ],
    },
}

STYLE_PRESET["yukari"]["rules"] += [
    "まとめ役。相手の長文を受けたときだけ短く要約する。",
]
STYLE_PRESET["yukari"]["surprise_examples"] = [
    "突然のポエム風たとえ", "昔話の一行ネタ", "静かなボケ"
]
STYLE_PRESET["yukari"]["emotion_bias"] = ["think","smile","neutral","shy"]

STYLE_PRESET["maki"]["rules"] += [
    "現実的な提案で話を前に進める係。",
    "勢いで脱線しても、1発で本筋へ戻す。"
]
STYLE_PRESET["maki"]["surprise_examples"] = [
    "急に行動宣言", "無茶な即決", "テンション高い誘い"
]
STYLE_PRESET["maki"]["emotion_bias"] = ["joy","tease","smile","laugh"]

STYLE_PRESET["ia"]["rules"] += [
    "唐突な思い出話や天然ボケを入れてもよい。ただし次で戻す。",
    "否定は避け、そっと代替案。"
]
STYLE_PRESET["ia"]["surprise_examples"] = [
    "小動物みたいな感想", "全然関係ない連想", "素直すぎる質問"
]
STYLE_PRESET["ia"]["emotion_bias"] = ["smile","laugh","shy","surprise"]

STYLE_PRESET["one"]["rules"] += [
    "逆張りやツンデレ指摘を時々入れる。",
    "言い過ぎたら1行で自己フォロー。"
]
STYLE_PRESET["one"]["surprise_examples"] = [
    "急な辛口ツッコミ", "逆方向の提案", "鋭い一言で流れを止める"
]
STYLE_PRESET["one"]["emotion_bias"] = ["smug","tease","annoyed","confuse"]


NAME_STYLE = {
    "yukari": {"self": "私", "others": {"maki": "マキさん", "ia": "IAちゃん", "one": "ONEさん"}},
    "maki":   {"self": "私", "others": {"yukari": "ゆかりちゃん", "ia": "IAちゃん", "one": "ONEちゃん"}},
    "ia":     {"self": "IA", "others": {"yukari": "ゆかりん", "maki": "マキ", "one": "ONE"}},
    "one":    {"self": "私", "others": {"yukari": "ゆかり", "maki": "マキ", "ia": "IA"}},
}

ALLOWED = ["yukari", "maki", "ia", "one"]

# 3D側で使いやすいように、軽量で被りにくい12種
EMOTION_CANON = [
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

# モデルが吐きがちな別表現を正規化
EMOTION_ALIAS = {
    "happy":"joy", "excited":"joy", "grin":"smile",
    "lol":"laugh", "giggle":"laugh",
    "sarcastic":"tease", "playful":"tease",
    "wow":"surprise", "shock":"surprise",
    "confused":"confuse", "puzzled":"confuse",
    "angry":"annoyed", "irritated":"annoyed",
    "bashful":"shy", "embarrassed":"shy",
    "proud":"smug", "confident":"smug",
    "thinking":"think", "ponder":"think",
    "tired":"sigh", "meh":"sigh",
}


# ====== SYSTEM生成 ======
def _style_block() -> str:
    lines = ["キャラクターごとの口調・語尾・相づち・追加ルール・意外性例・感情傾向："]
    for name in ALLOWED:
        s = STYLE_PRESET[name]
        ends = " / ".join(s["endings"])
        fills = " / ".join(s["fillers"])
        rules = " ".join("・"+r for r in s["rules"])
        surprises = " / ".join(s.get("surprise_examples", []))
        emo_bias = " / ".join(s.get("emotion_bias", []))
        lines.append(
            f"- {name}: tone={s['tone']}｜endings={ends}｜fillers={fills}｜"
            f"rules: {rules}｜surprises: {surprises}｜emotion_bias: {emo_bias}"
        )
    lines.append("\n呼び名・一人称ルール：")
    for name in ALLOWED:
        ns = NAME_STYLE[name]
        others = " / ".join([f"{k}→{v}" for k, v in ns["others"].items()])
        lines.append(f"- {name}: 一人称='{ns['self']}'｜他人称={others}")
    lines.append("\n使用可能な emotion 候補：" + ", ".join(EMOTION_CANON))
    lines.append("各発話は emotion を1つ付与。各キャラの emotion_bias を2/3の確率で優先し、残りは自由。")
    lines.append("ゆかりのみ『ですね』使用可。他キャラは『ですね』禁止。翻訳調/説明調を避け、短く直感的に。")
    return "\n".join(lines)


def build_system_prompt() -> str:
    return (
        "あなたは女子高生4人組（結月ゆかり・マキ・IA・ONE）の会話スクリプトを作成します。\n"
        "出力は**次のJSONのみ**。説明文・装飾・前置きは禁止。\n"
        "{ \"turns\": [ {\"speaker\":\"yukari|maki|ia|one\", \"text\":\"...\", \"emotion\":\"...\"} ] }\n"
        "会話ルール：\n"
        "- surprise_min と derail_ratio の指示に従う。意外性の例：急な個人体験、逆張り、一瞬の別案、ちょいボケ。\n"
        "- キャラの役割：ゆかり=まとめ役、マキ=前進係、IA=天然/連想、ONE=逆張り/鋭い指摘。\n"
        "- 脱線したら次の一言か二言で必ず話題に戻す（『で、話戻すけど…』など）。\n"
        "- 末尾の語尾は各キャラのendingsを20%程度の確率で使う。ただし多用しない。\n"
        "- 最初の1行は USER から指定されたキャラが、トピックを自分の発言として言い直す。\n"
        "- 言い直しは『体験談』『感情』『愚痴』『ちょっとした提案』など自然なセリフにし、要約・解説は禁止。\n"
        "- 4人全員が最低1回は発言。順番は固定しない。\n"
        "- 1〜2文/発話。直前の発話に具体的に反応する。会話が止まると判断した場合、その後に軽い新情報や質問を足す。\n"
        "- 一定の割合で、具体的な固有名詞などをネットから拾って付け足す。\n"
        "- 脱線・冗談・遮りも少しOK。女子高生らしい軽さで。\n"
        "- 最後の2〜3行で小さな区切り（次やること/保留/締めなど）。ただし会話がとりとめなくなってもOK\n"
        "- 誰かの意見・質問をスルーしてないこと。特に怪我や事故、トラブルなど驚く事象についてはきちんと相手を心配すること。\n"
        "- 会話全体で複数回の意外性（逆張り/ボケ/個人体験）を入れるが、流れは最終的に前進させる。\n"
        "- 自分の発言に一貫性を保っているかなどを最後にチェックすること。\n"
        "- 日本語としての助詞が違和感がないかを最後にチェックすること。ただし女子高生としての崩し方の範囲内であればOK。\n"
    ) + "\n" + _style_block()

# ====== USER生成 ======
def build_user(topic: str, turns: int) -> str:
    opener = random.choice(ALLOWED)
    surprise = 2
    derail_ratio = 0.25  # 4分の1くらいを脱線・ボケ許容
    # 役割ごとにサプライズ加点（ONEとIAに多め）
    role_bias = {"one":{"surprise_bonus":1}, "ia":{"surprise_bonus":1}, "maki":{"surprise_bonus":0}, "yukari":{"surprise_bonus":0}}

    return (
        f"topic={topic}\n"
        f"turns={turns}\n"
        f"最初の1行は {opener} が担当する。{opener} は指定トピックを『自分の体験・感情のこもった自然な一言』として言い直す。"
        "要約・解説は禁止。\n"
        f"surprise_min={surprise}\n"
        f"derail_ratio={derail_ratio}\n"
        f"role_bias={role_bias}\n"
        "必ずJSONのみ返答。JSON以外の文字は一切含めないこと。"
    )


# ====== 簡易サニタイズ：モデルが万一JSON以外を返した時の保険 ======
JSON_HEAD = re.compile(r"\{\s*\"turns\"\s*:", re.DOTALL)

def _extract_json_block(text: str) -> str | None:
    """テキストから先頭の { "turns": ... } ブロックを抜く簡易版"""
    m = JSON_HEAD.search(text)
    if not m:
        return None
    start = m.start()
    # 括弧バランスで末尾を探す
    depth, i = 0, start
    while i < len(text):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i+1]
        i += 1
    return None

# ====== 生成本体 ======
def generate_turns(topic: str, turns: int, model: str = "gpt-4o-mini", temperature: float = 1.15) -> List[Dict[str, Any]]:
    system = build_system_prompt()
    user = build_user(topic, turns)

    resp = client.responses.create(
        model=model,
        temperature=temperature,
        max_output_tokens=2048,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        top_p=0.9,
    )

    txt = (resp.output_text or "").strip()

    # JSONとして読む
    try:
        data = json.loads(txt)
        turns_data = data.get("turns", [])
        return _postfilter(turns_data, turns)
    except Exception:
        pass

    # { "turns": ... } 抜き出し
    block = _extract_json_block(txt)
    if block:
        try:
            data = json.loads(block)
            turns_data = data.get("turns", [])
            return _postfilter(turns_data, turns)
        except Exception:
            pass

    print("[llm_openai] JSON decode failed. raw_head:", txt[:400])
    return []


# ====== 後処理：型の安全化・行数合わせ・speaker正規化・emotionデフォルト ======
def _norm_speaker(x: str) -> str:
    x = (x or "").strip().lower()
    if x in ALLOWED:
        return x
    # ゆるい正規化
    if "yukari" in x or "ゆかり" in x: return "yukari"
    if "maki" in x or "マキ" in x:     return "maki"
    if x == "ia" or "ia" in x:         return "ia"
    if "one" in x or "ワン" in x:      return "one"
    return random.choice(ALLOWED)

def _postfilter(turns_data: List[Dict[str, Any]], want: int) -> List[Dict[str, Any]]:
    cleaned = []
    for t in turns_data:
        spk = _norm_speaker(t.get("speaker", ""))
        text = (t.get("text") or "").strip()
        emo_raw = (t.get("emotion") or "neutral").strip().lower()
        if not text:
            continue
        emo = EMOTION_ALIAS.get(emo_raw, emo_raw)
        if emo not in EMOTION_CANON:
            emo = "neutral"
        cleaned.append({"speaker": spk, "text": text, "emotion": emo})

    if len(cleaned) > want:
        cleaned = cleaned[:want]
    return cleaned

