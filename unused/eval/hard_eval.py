import json, random
from app.llm_client import LLMClient

TESTS = [
    {"q": "8454÷140080", "a": "0.0603 付近 (小数第4位)"},
    {"q": "Windowsサービスが即停止する時の基本手順", "a": "イベントログ/依存関係/権限…"},
]

llm = LLMClient()

def ask(q: str, n: int = 3) -> str:
    messages = [{"role": "user", "content": q}]
    outs = [llm.chat(messages, temperature=0.7) for _ in range(n)]
    # もっとも多数派の要約を返す（雑だが強い）
    best = max(set(outs), key=outs.count)
    return best

if __name__ == "__main__":
    for t in TESTS:
        out = ask(t["q"], n=5)
        print("Q:", t["q"])
        print("A:", out[:300], "\n")
        