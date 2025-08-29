import os
import glob
import re
from typing import List, Tuple
from rank_bm25 import BM25Okapi

TOKEN_SPLIT = re.compile(r"\w+|[^\w\s]", re.UNICODE)

class RAGStore:
    def __init__(self, docs_dir: str):
        self.docs_dir = docs_dir
        self.paths: List[str] = []
        self.docs: List[str] = []
        self.tokens: List[List[str]] = []
        self.bm25 = None

    def _tokenize(self, text: str) -> List[str]:
        return [t.lower() for t in TOKEN_SPLIT.findall(text)]

    def ingest(self):
        self.paths = []
        self.docs = []
        for p in glob.glob(os.path.join(self.docs_dir, "**/*"), recursive=True):
            if os.path.isdir(p):
                continue
            if os.path.splitext(p)[1].lower() in {".txt", ".md"}:
                with open(p, "r", encoding="utf-8", errors="ignore") as f:
                    txt = f.read()
                self.paths.append(p)
                self.docs.append(txt)
        self.tokens = [self._tokenize(t) for t in self.docs]
        if self.tokens:
            self.bm25 = BM25Okapi(self.tokens)

    def search(self, query: str, k: int = 5) -> List[Tuple[str, str]]:
        if not self.bm25:
            return []
        q = self._tokenize(query)
        scores = self.bm25.get_scores(q)
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self.paths[i], self.docs[i]) for i in top_idx]
    