"""指定版の話者エンコーダーだけを取得・検証する。利用者の音声は送信しない。"""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import tempfile
import urllib.request


MODEL_SHA256 = "39373b86598fa3da9fcddee6142382efe09777e8d37dc9c0561f41f0070f134e"
MODEL_URL = "https://raw.githubusercontent.com/WUSTL-CSPL/AntiFake/0eb55a12589ccf8480acef67ea2beec46bea15f3/saved_models/default/encoder.pt"
MODEL_PATH = Path(__file__).resolve().parent / "encoder.pt"
MODEL_MAX_BYTES = 20 * 1024 * 1024


def digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def main() -> int:
    if MODEL_PATH.is_file() and digest(MODEL_PATH) == MODEL_SHA256:
        print("モデルを確認しました。再取得は不要です。")
        return 0
    print("音声モデル（約 17 MB）を公式リポジトリーから取得しています…")
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=MODEL_PATH.parent, prefix="encoder-", suffix=".part", delete=False) as output:
            temporary_path = Path(output.name)
            request = urllib.request.Request(MODEL_URL, headers={"User-Agent": "AntiFake-JP-local-prototype/0.1"})
            with urllib.request.urlopen(request, timeout=60) as response:
                size = 0
                while block := response.read(1024 * 1024):
                    size += len(block)
                    if size > MODEL_MAX_BYTES:
                        raise ValueError("取得ファイルが想定サイズを超えています。")
                    output.write(block)
        if digest(temporary_path) != MODEL_SHA256:
            raise ValueError("モデルの検証に失敗しました。保存済みのモデルは変更していません。")
        temporary_path.replace(MODEL_PATH)
        temporary_path = None
        print("モデルの取得と検証が完了しました。")
        return 0
    except Exception as error:
        print(f"モデルを取得できませんでした: {error}", file=sys.stderr)
        return 1
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
