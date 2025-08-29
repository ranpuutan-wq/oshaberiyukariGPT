import queue, threading, simpleaudio as sa, tempfile, os

class AudioQueue:
    def __init__(self):
        self.q = queue.Queue()
        self.th = threading.Thread(target=self._worker, daemon=True)
        self.th.start()

    def _play_bytes(self, data: bytes):
        # simpleaudioの互換事情で一旦ファイルに落として再生
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(data)
            path = f.name
        try:
            wave = sa.WaveObject.from_wave_file(path)
            play = wave.play()
            play.wait_done()
        finally:
            try: os.remove(path)
            except: pass

    def _worker(self):
        while True:
            data = self.q.get()
            try:
                self._play_bytes(data)
            finally:
                self.q.task_done()

    def enqueue(self, data: bytes):
        self.q.put(data)

    def wait_all(self):
        self.q.join()
