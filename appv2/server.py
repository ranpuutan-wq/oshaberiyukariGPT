# app/server.py（例）
from fastapi import FastAPI
from pydantic import BaseModel
from app.llm_openai import generate_turns

app = FastAPI()

class GenReq(BaseModel):
    topic: str
    turns: int = 12
    model: str | None = None

@app.post("/generate_4")
def generate_4(req: GenReq):
    model = req.model or "gpt-4.1"   # or "gpt-4o-mini"
    turns = max(4, min(64, req.turns))
    turns_data = generate_turns(req.topic, turns, model=model)
    return {"turns": turns_data}
