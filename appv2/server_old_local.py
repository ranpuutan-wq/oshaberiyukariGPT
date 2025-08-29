import yaml, json, uuid, random
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from .llm_client import LLMClient
from .rag import RAGStore
from .tools import Tools
from .agent import Agent
from .schemas import Ask, GenerateReq, GenerateResp, Turn
from .prompt_multi import build_4p_messages, ALLOWED

app = FastAPI()

with open("./app/config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

rag = RAGStore(cfg["rag"]["docs_dir"]); rag.ingest()
llm = LLMClient(cfg["model"].get("base_url"), cfg["model"].get("name"))
tools = Tools(rag)
agent = Agent(llm, tools, max_tool_calls=cfg["agent"]["max_tool_calls"])

@app.post("/ask")
async def ask(req: Ask):
    prompt = (
        "あなたはフレンドリーで自然体な日本語を話す女の子キャラです。"
        "敬語ではなく、日常会話っぽいラフな口調で答えてください。\n"
        f"ユーザー: {req.query}"
    )
    answer = agent.run(prompt) or "(no output)"
    return JSONResponse(content={"answer": answer}, media_type="application/json; charset=utf-8")

def parse_jsonl(raw: str, expected: int):
    items = []
    for line in raw.strip().splitlines():
        try:
            obj = json.loads(line)
            spk = obj.get("speaker", "")
            txt = obj.get("text", "")
            emo = obj.get("emotion", "neutral")
            if spk in ALLOWED and txt:
                items.append({"speaker": spk, "text": txt, "emotion": emo})
        except json.JSONDecodeError:
            continue
    return items[:expected]

def rotate_fix(items, expected):
    # 受け取れたテキストを順番固定で並べ替える（不足は空行除外のまま）
    out = []
    order = ["yukari","maki","ia","one"]
    i = 0
    for n in range(expected):
        spk = order[n % 4]
        if i < len(items):
            out.append({"speaker": spk, "text": items[i]["text"], "emotion": items[i]["emotion"]})
            i += 1
        else:
            out.append({"speaker": spk, "text": "…", "emotion": "neutral"})
    return out

@app.post("/generate_4", response_model=GenerateResp)
async def generate_4(req: GenerateReq):
    turns = max(4, min(32, req.turns or 12))
    msgs = build_4p_messages(topic=req.topic, turns=turns)

    # 1回目生成（温度低めでフォーマット安定）
    raw = llm.chat(msgs, temperature=0.4)
    parsed = parse_jsonl(raw, turns)

    # フォーマット不良や独り語りが多ければ、1度だけ再生成
    if len(parsed) < turns // 2:
        raw = llm.chat(msgs, temperature=0.8)
        parsed = parse_jsonl(raw, turns)

    # 最終的に順番を強制整形
    fixed = rotate_fix(parsed, turns)

    resp_turns = [
        Turn(id=str(uuid.uuid4()), speaker=t["speaker"], text=t["text"], emotion=t["emotion"])
        for t in fixed
    ]
    return GenerateResp(topic=req.topic, turns=resp_turns)

