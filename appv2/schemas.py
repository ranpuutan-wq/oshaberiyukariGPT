from pydantic import BaseModel
from typing import List, Optional

class Ask(BaseModel):
    query: str

class GenerateReq(BaseModel):
    topic: str
    turns: int = 12
    seed: Optional[int] = None

class Turn(BaseModel):
    id: str
    speaker: str
    text: str
    emotion: str = "neutral"

class GenerateResp(BaseModel):
    topic: str
    turns: List[Turn]
    