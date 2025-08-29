import math
import json
import traceback
from typing import Dict, Any, List

from .rag import RAGStore

class ToolError(Exception):
    pass

class Tools:
    def __init__(self, rag: RAGStore):
        self.rag = rag

    def list_tools(self) -> List[Dict[str, Any]]:
        return [
            {"name": "search_docs", "desc": "社内/ローカル文書をBM25で検索し、該当箇所を返す。args: {query:str}"},
            {"name": "calc", "desc": "安全な数式評価（+ - * / ** () のみ）。args: {expr:str}"},
            {"name": "run_python", "desc": "短いPythonコードをサンドボックス実行。print出力のみ返す。args: {code:str}"},
        ]

    def _safe_eval(self, expr: str) -> str:
        allowed = {"__builtins__": None}
        # 極力安全側（eval対象は算術のみ推奨）
        try:
            val = eval(expr, allowed, {})
        except Exception as e:
            raise ToolError(str(e))
        return str(val)

    def search_docs(self, args: Dict[str, Any]) -> str:
        q = str(args.get("query", ""))
        hits = self.rag.search(q, k=5)
        snippets = []
        for path, doc in hits:
            snippet = doc[:800].replace("\n", " ")
            snippets.append({"path": path, "snippet": snippet})
        return json.dumps(snippets, ensure_ascii=False)

    def calc(self, args: Dict[str, Any]) -> str:
        expr = str(args.get("expr", ""))
        return self._safe_eval(expr)

    def run_python(self, args: Dict[str, Any]) -> str:
        code = str(args.get("code", ""))
        if len(code) > 1000:
            raise ToolError("code too long")
        # 超簡易サンドボックス
        env = {"__builtins__": {"print": print, "range": range}}
        out_lines: List[str] = []
        def cap_print(*a, **kw):
            out_lines.append(" ".join(map(str, a)))
        env["print"] = cap_print
        try:
            exec(code, env, {})
        except Exception:
            return "ERROR:\n" + traceback.format_exc(limit=2)
        return "\n".join(out_lines) if out_lines else "(no output)"

    def call(self, name: str, args: Dict[str, Any]) -> str:
        if name == "search_docs":
            return self.search_docs(args)
        if name == "calc":
            return self.calc(args)
        if name == "run_python":
            return self.run_python(args)
        raise ToolError(f"unknown tool: {name}")
    