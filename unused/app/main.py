# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.server import app as gen_app       # /generate_4 等
from app.server_tts import app as tts_app   # /speak

app = FastAPI(title="AI4人チャットAPI (all-in-one)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 必要なら絞る
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 生成系は /gen 配下
app.mount("/gen", gen_app)
# TTS は /tts 配下
app.mount("/tts", tts_app)

@app.get("/health")
def health():
    return {"ok": True}
