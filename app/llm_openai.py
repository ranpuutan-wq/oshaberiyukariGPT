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

# ---- その1: 正規化マップ ----
CANON = {
    "結月ゆかり": "yukari", "ゆかり": "yukari", "yukari": "yukari",
    "弦巻マキ": "maki", "マキ": "maki", "maki": "maki",
    "IA": "ia", "ia": "ia", "イア": "ia",
    "ONE": "one", "one": "one", "オネ": "one",
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

# ====== propose(一言生成) 用 追加関数 ======

def build_propose_system_prompt() -> str:
    """
    司会用（1セリフだけ生成）の SYSTEM プロンプト。
    既存の STYLE_PRESET / NAME_STYLE / EMOTION_CANON を踏襲。
    - 出力は JSON オブジェクト1件のみ（余計な文字禁止）
    - line_rubyは、lineの アルファベット字のみをカタカナ読みに変換して line_ruby として必ず返す
    - emotion は EMOTION_CANON のいずれか
    - topic は入力と同じ値または、変更を行った場合は変更後の値のどちらかを必ず出力
    """
    rules = [
        "あなたは4人会話( yukari, maki, ia, one )の司会として、主題から逸脱しない1セリフだけを提案する。",
        "出力は JSON オブジェクト1件のみ。説明/前置き/コードブロックは禁止。",
        "speaker は ['yukari','maki','ia','one'] のいずれか。",
        "キャラの役割：ゆかり=まとめ役/フォロー、マキ=前進係、IA=天然/話題変えたがる、ONE=逆張り/鋭い指摘。"
        "emotion は既定の候補から1つだけ選ぶ（後述の候補）。",
        "line は“自然な口語”で 1～2文。パターンは「feelingsの自分の感情と、直前の他者セリフを元に反応」または「関連する別の話題を提示(◯◯と言えばこないださぁ・・等)」",
        "line_ruby は line をもとに、ひらがな・カタカナは同じ内容を設定し、アルファベット文字はカタカナに変換。 ）",
        "【重要】出力JSONには 'topic' キーを必ず含める（必須）。"
        "【重要】発言の内容に応じて、topicを変更し、JSONに出力する。"
        "【重要】50%の確率で、topicを強制的に変更しJSONに出力する。(例： 「全然関係ないけど・・・」)"
        "【重要】response_format=json_object で厳密JSONのみを返すこと。キー欠落は禁止。"
        "各キャラの口調・語尾・相づち・役割・emotion傾向は下記定義に従う。",
        "回答は短く、直感的に。翻訳調/説明調を避ける。脱線・冗談・遮りも少しOK。",
        "怪我や事故、トラブルなどの話題に対しては相手を心配する。",
        "30%の割合で、具体的な固有名詞をネットから拾って付け足す。",
        "20%の確率で意外性（逆張り/ボケ/個人体験）を入れる",
        "女子高生らしい軽さ。翻訳調/説明調/冗長を避け、砕けた文体を許容する。",
        
    ]
    # 既存の定義を埋め込む
    lines = []
    lines.append("\n".join(rules))
    lines.append("\n--- キャラクター定義（STYLE_PRESET由来） ---")
    lines.append(_style_block())  # 既存の補助関数を再利用
    lines.append("\n--- 呼び名・一人称（NAME_STYLE由来） ---")
    for name in ALLOWED:
        ns = NAME_STYLE[name]
        others = " / ".join([f"{k}→{v}" for k, v in ns["others"].items()])
        lines.append(f"- {name}: 一人称='{ns['self']}'｜他人称={others}")

    lines.append("\n--- 使用可能な emotion 候補 ---")
    lines.append(", ".join(EMOTION_CANON))

    lines.append(
        "\n--- 出力仕様(JSON) ---\n"
        "{\n"
        '  "speaker": "yukari|maki|ia|one",\n'
        '  "line": "str",\n'
        '  "line_ruby": "str (英字必ずカタカナに変換)",\n'
        '  "emotion": "one of EMOTION_CANON",\n'
        '  "feelings": {"yukari":"str","maki":"str","ia":"str","one":"str"},\n'
        '  "topic": "str"\n'
        "}\n"
        "※ feelings は『主題に対する今の感じ方』を各キャラ分、短く更新する。"
    )
    #print ("[DEBUG] lines ", lines)
    return "\n".join(lines)


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

def build_propose_user_prompt(
    topic: str,
    history: list[dict],
    speakers: list[str] | None = None,
    prior_feelings: dict | None = None,
) -> str:
    """
    USER プロンプト。方法1の入力仕様に対応：
      - 会話の主題 (topic)
      - 直前の他者セリフ (history の末尾)
      - 4人の感じたこと（前回時点） prior_feelings
    """
    speakers = speakers or ALLOWED
    last_utt = history[-1]["text"] if history else ""

    payload = {
        "topic": topic,
        "speakers": speakers,
        "last_utterance_raw": last_utt,
        "prior_feelings": prior_feelings or {},
        # コンテキストは直近6件程度までに抑えてトークン効率化
        "history_tail": history[-6:],
        "output_schema": {
            "speaker": "one of ['yukari','maki','ia','one']",
            "line": "str",
            "line_ruby": "str (English word top katakana)",
            "emotion": "str among EMOTION_CANON",
            "feelings": {"yukari":"str","maki":"str","ia":"str","one":"str"},
            "topic": "str",

        },
        # 生成の微調整ヒント
        "hints": {
            "stay_on_topic": True,
            "keep_flowing": False,
            "length": "1 sentences",
        },
    }
    # LLMに渡すときは最終的に文字列化（Responses互換のため）
    return str(payload)


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


# ====== 司会用（1セリフ生成） ======
def build_propose_messages(
    topic: str,
    history: list[dict],
    speakers: list[str] | None = None,
    prior_feelings: dict | None = None,
) -> list[dict]:
    """
    Chat Completions/Responses の messages 配列を返す。
    - system: build_propose_system_prompt()
    - user  : build_propose_user_prompt(...)
    """
    return [
        {"role": "system", "content": build_propose_system_prompt()},
        {"role": "user", "content": build_propose_user_prompt(topic, history, speakers, prior_feelings)},
    ]


def propose_one_line(
    topic: str,
    history: list[dict],
    speakers: list[str] | None = None,
    prior_feelings: dict | None = None,
    *,
    model: str | None = None,
    temperature: float = 0.7,
) -> dict:
    """
    実行ヘルパ（任意）：LLM を叩いて 1セリフの JSON を返す。
    既存 client(OpenAI) を使い、response_format='json_object' で安全化。
    """
    msgs = build_propose_messages(topic, history, speakers, prior_feelings)
    mdl = model or os.getenv("GEN_MODEL_ONE", "gpt-4.1-mini")
    rsp = client.chat.completions.create(
        model=mdl,
        messages=msgs,
        response_format={"type": "json_object"},
        temperature=temperature,
    )
    import json
    content = (rsp.choices[0].message.content or "").strip()
    try:
        obj = json.loads(content)
    except Exception:
        # 最低限のフォールバック
        obj = {"speaker":"yukari","line":content,"line_ruby":content,"emotion":"neutral",
               "feelings":{"yukari":"","maki":"","ia":"","one":""},"topic":topic}
    # スピーカ正規化（既存の normalize_spk があれば使う）
    try:
        obj["speaker"] = normalize_spk(obj.get("speaker"))
    except Exception:
        pass
    return obj

def normalize_spk(s: str) -> str:
    if not s:
        return "yukari"
    return CANON.get(s.strip(), s.strip().lower())
