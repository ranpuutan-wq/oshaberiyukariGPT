import json
from typing import List, Dict
from .llm_client import LLMClient
from .tools import Tools, ToolError

SYSTEM_PROMPT = (
    "あなたは日本語で会話する有能なエージェントです。必要があればツールを呼び出してから最終回答を返します。\n"
    "ツールを使う場合は必ず次の形式で出力してください。\n\n"
    "TOOL_CALL {\"name\": <tool_name>, \"args\": { ... }}\n\n"
    "回答を出すときは次の形式で出力してください。\n\nFINAL <answer>\n\n"
    "ツール一覧はユーザーが求めるときだけ簡潔に示してください。"
)

TOOL_HINT = (
    "使用可能ツール:\n"
    "- search_docs(query:str) -> JSON スニペット\n"
    "- calc(expr:str) -> 数式評価\n"
    "- run_python(code:str) -> print出力\n"
)

class Agent:
    def __init__(self, llm: LLMClient, tools: Tools, max_tool_calls: int = 5):
        self.llm = llm
        self.tools = tools
        self.max_tool_calls = max_tool_calls

    def run(self, user_input: str) -> str:
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": TOOL_HINT + "\nユーザー: " + user_input},
        ]

        for _ in range(self.max_tool_calls):
            out = self.llm.chat(messages)
            out = out.strip()
            if out.startswith("TOOL_CALL"):
                json_part = out[len("TOOL_CALL"):].strip()
                try:
                    spec = json.loads(json_part)
                    name = spec.get("name")
                    args = spec.get("args", {})
                    tool_result = self.tools.call(name, args)
                    messages.append({"role": "assistant", "content": out})
                    messages.append({"role": "tool", "content": tool_result})
                except (json.JSONDecodeError, ToolError) as e:
                    messages.append({"role": "assistant", "content": out})
                    messages.append({"role": "tool", "content": f"ERROR: {e}"})
                    continue
            elif out.startswith("FINAL"):
                return out[len("FINAL"):].strip()
            else:
                # モデルが素で回答したら、そのまま返す
                return out
        return "(ツール呼び出し回数上限に達しました)"
    