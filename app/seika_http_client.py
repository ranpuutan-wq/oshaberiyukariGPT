# seika_http_client.py（新規ファイルでOK）
import time, os, requests, json
from pathlib import Path

class SeikaClient:
    def __init__(self, host="127.0.0.1", port=7180, user="SeikaServerUser", password="SeikaServerPassword"):
        self.base = f"http://{host}:{port}"
        self.auth = (user, password)  # Basic相当。AssistSeikaはこれで受ける設定が多い

    def _is_audio(self, r: requests.Response) -> bool:
        ct = r.headers.get("Content-Type", "")
        return ct.startswith("audio/") or ct in ("application/octet-stream",)

    # --- 1) 直接バイトを返す系（もし対応していればこれが最速）
    def tts_bytes_try(self, cid: int, text: str) -> bytes | None:
        # 例: /AVATOR2/TEXTTOWAVE?cid=2102&text=... など
        url = f"{self.base}/AVATOR2/TEXTTOWAVE"
        r = requests.post(url, data={"cid": cid, "text": text}, auth=self.auth, timeout=120)
        if r.status_code != 200:
            return None
        if self._is_audio(r):
            return r.content
        # JSONっぽい場合はWAVを返していない
        return None

    # --- 2) ファイル保存モード：サーバにWAVを書かせてから取得
    def tts_to_file_then_load(self, cid: int, text: str, out_dir: str, filename: str | None=None, delete_after=True) -> bytes:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if not filename:
            import time
            filename = f"seika_{cid}_{int(time.time()*1000)}.wav"
        out_path = (out_dir / filename).resolve()

        # 例: /AVATOR2/SAVEWAVE に JSON で {cid, text, path} を渡す系
        url = f"{self.base}/AVATOR2/SAVEWAVE"
        payload = {"cid": cid, "text": text, "path": str(out_path)}
        r = requests.post(url, json=payload, auth=self.auth, timeout=120)
        r.raise_for_status()

        # 出力完了を待つ（小さすぎる＝まだ書き込み途中のことがある）
        for _ in range(100):
            if out_path.exists() and out_path.stat().st_size > 2000:  # だいたい2KB超えたら中身あり
                data = out_path.read_bytes()
                if delete_after:
                    try: out_path.unlink()
                    except: pass
                return data
            time.sleep(0.05)
        raise RuntimeError(f"WAVが出力されませんでした: {out_path}")

    def speak_or_get(self, cid: int, text: str, docroot: str) -> bytes:
        # まずは「直接バイト返す」系にトライ
        b = self.tts_bytes_try(cid, text)
        if b: 
            return b
        # ダメなら「保存→読み出し」にフォールバック
        return self.tts_to_file_then_load(cid, text, out_dir=docroot)
