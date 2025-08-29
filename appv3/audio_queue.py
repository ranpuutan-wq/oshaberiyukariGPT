# --- 先頭に追加（talk_runner.py） ---
import asyncio, concurrent.futures, queue, threading, io, wave
import numpy as np

class AudioQueue:
    def __init__(self):
        self.q = queue.Queue()
        self.t = threading.Thread(target=self._worker, daemon=True)
        self.t.start()
    def _worker(self):
        while True:
            wav_bytes, delay_ms, gain_db = self.q.get()
            try:
                if delay_ms:
                    time.sleep(delay_ms/1000.0)
                if gain_db:
                    wav_bytes = apply_gain(wav_bytes, gain_db)
                play_bytes(wav_bytes)
            except Exception as e:
                print("[audio][err]", e)
            finally:
                self.q.task_done()

QUEUES = {}
def enqueue_audio(speaker: str, wav_bytes: bytes, delay_ms=0, gain_db=0):
    if not wav_bytes:
        return
    if speaker not in QUEUES:
        QUEUES[speaker] = AudioQueue()
    QUEUES[speaker].q.put((wav_bytes, delay_ms, gain_db))

def apply_gain(wav_bytes: bytes, gain_db: float) -> bytes:
    try:
        bio = io.BytesIO(wav_bytes)
        with wave.open(bio, 'rb') as wf:
            params = wf.getparams()
            if params.sampwidth != 2:
                return wav_bytes
            frames = wf.readframes(params.nframes)
        arr = np.frombuffer(frames, dtype=np.int16).astype(np.float64)
        scale = 10 ** (gain_db / 20.0)
        arr = np.clip(arr * scale, -32768, 32767).astype(np.int16)
        out = io.BytesIO()
        with wave.open(out, 'wb') as wfo:
            wfo.setparams(params)
            wfo.writeframes(arr.tobytes())
        return out.getvalue()
    except Exception:
        return wav_bytes