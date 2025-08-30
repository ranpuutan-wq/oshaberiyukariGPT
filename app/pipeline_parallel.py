# app/pipeline_parallel.py
from __future__ import annotations
import asyncio
from dataclasses import dataclass
from typing import Optional, Callable, Awaitable, List, Dict, Any

@dataclass(frozen=True)
class SpeechItem:
    idx: int
    speaker: str
    text: str
    emotion: Optional[str] = None

@dataclass(frozen=True)
class TTSDone:
    idx: int
    speaker: str
    wav: bytes
    can_overlap: bool
    gain_db: float

class TTSPipeline:
    """
    話者ごと直列・話者間は並列でTTSを回し、結果（TTSDone）を返すだけの軽量パイプラインだッピ。
    再生タイミングの最終決定は既存 PlaybackScheduler に任せる（順序とオーバーラップはそこで担保）。
    """
    def __init__(
        self,
        tts_func: Callable[[str, str, Optional[str]], Awaitable[bytes]],
        *,
        max_parallel_tts: int = 3,
        per_speaker_serial: bool = True,
    ) -> None:
        self.tts_func = tts_func
        self.max_parallel_tts = max_parallel_tts
        self.per_speaker_serial = per_speaker_serial
        self._sem = asyncio.Semaphore(self.max_parallel_tts)
        self._speaker_locks: Dict[str, asyncio.Lock] = {}

    def _lock_for(self, spk: str) -> asyncio.Lock:
        if spk not in self._speaker_locks:
            self._speaker_locks[spk] = asyncio.Lock()
        return self._speaker_locks[spk]

    async def _tts_one(self, item: SpeechItem) -> TTSDone:
        lock = self._lock_for(item.speaker) if self.per_speaker_serial else None
        async with self._sem:
            if lock:
                async with lock:
                    wav = await self.tts_func(item.speaker, item.text, item.emotion)
            else:
                wav = await self.tts_func(item.speaker, item.text, item.emotion)

        # 20文字以下の短文は “被せ”候補（既存talk_runnerのポリシーに合わせる）
        can_overlap = (len(item.text) <= 20 and item.idx > 0)
        gain_db = -4 if can_overlap else 0.0
        return TTSDone(idx=item.idx, speaker=item.speaker, wav=wav,
                       can_overlap=can_overlap, gain_db=gain_db)

    async def run_collect(self, items: List[SpeechItem]) -> List[TTSDone]:
        tasks = [asyncio.create_task(self._tts_one(it)) for it in items]
        outs: List[TTSDone] = []
        for t in asyncio.as_completed(tasks):
            outs.append(await t)
        outs.sort(key=lambda x: x.idx)  # “会話の順序”で返す
        return outs
