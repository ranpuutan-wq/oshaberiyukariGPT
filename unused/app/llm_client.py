import os
import requests
from typing import List, Dict, Any

class LLMClient:
    def __init__(self, base_url: str | None = None, model: str | None = None):
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model = model or os.getenv("OLLAMA_MODEL", "llama3:8b-instruct")

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.2) -> str:
        # --- 1) まず /api/chat を試す ---
        try:
            url = f"{self.base_url}/api/chat"
            payload: Dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": temperature}
            }
            r = requests.post(url, json=payload, timeout=120)
            r.raise_for_status()
            data = r.json() or {}
            text = (data.get("message", {}) or {}).get("content", "")
            if text and text.strip():
                return text
        except Exception:
            pass  # フォールバックへ

        # --- 2) ダメなら /api/generate にフォールバック ---
        #    （多くのモデルは generate なら安定）
        prompt = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)
        url = f"{self.base_url}/api/generate"
        payload = {"model": self.model, "prompt": prompt, "stream": False, "options": {"temperature": temperature}}
        r = requests.post(url, json=payload, timeout=120)
        r.raise_for_status()
        data = r.json() or {}
        return data.get("response", "") or ""
    