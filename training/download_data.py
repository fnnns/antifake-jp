"""固定した公開アーカイブを取得し、学習用データの検証記録を保存します。"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time
import urllib.request

URL = "https://huggingface.co/tanaka5/models0918/resolve/9286a16c840cd74a1e73d645678f6ce31fc91abd/jvs_ver1.zip"
EXPECTED_BYTES = 3536595425
EXPECTED_SHA256 = "37180e2f87bd1a3e668d7c020378f77cebf61dd57d4d74c71eb0114f386a3999"
HISTORICAL_MD5 = "2987778b0ee830914bfebb97783d0c3e"


def verify(path: Path):
    """全体の検証値を計算し、固定したファイルとの一致を確認します。"""
    if path.stat().st_size != EXPECTED_BYTES:
        raise ValueError("アーカイブのサイズが一致しません。")
    sha, md5 = hashlib.sha256(), hashlib.md5()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            sha.update(block)
            md5.update(block)
    if sha.hexdigest() != EXPECTED_SHA256:
        raise ValueError("アーカイブの SHA-256 が一致しません。取得済みファイルは保持しています。")
    return {"sha256": sha.hexdigest(), "md5": md5.hexdigest(),
            "historical_md5_matches": md5.hexdigest() == HISTORICAL_MD5}


def main():
    parser = argparse.ArgumentParser(description="JVS 形式の公開アーカイブを取得・検証します。")
    parser._optionals.title = "オプション"
    parser.add_argument("--directory", type=Path, default=Path(__file__).resolve().parent / "raw", help="保存先")
    parser.add_argument("--verify-only", action="store_true", help="取得せず、保存済みファイルを検証します。")
    args = parser.parse_args()
    directory = args.directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    final = directory / "jvs_ver1.zip"
    partial = directory / "jvs_ver1.zip.part"
    if not final.exists():
        if args.verify_only:
            raise FileNotFoundError("検証するアーカイブがありません。")
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > EXPECTED_BYTES:
            raise ValueError("途中ファイルが想定より大きくなっています。別の保存先を指定してください。")
        if shutil.disk_usage(directory).free < EXPECTED_BYTES - offset + 3 * 1024 ** 3:
            raise ValueError("学習用特徴量とアーカイブを保存するための空き容量が足りません。")
        if offset < EXPECTED_BYTES:
            request = urllib.request.Request(URL, headers={"User-Agent": "AntiFake-JP-local-research/0.2"})
            if offset:
                request.add_header("Range", f"bytes={offset}-")
            with urllib.request.urlopen(request, timeout=60) as response:
                if offset and (response.status != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-")):
                    raise ValueError("途中からの取得に対応していません。途中ファイルは保持しています。")
                total = offset
                last_log = 0.0
                with partial.open("ab" if offset else "wb") as stream:
                    while block := response.read(4 * 1024 * 1024):
                        total += len(block)
                        if total > EXPECTED_BYTES:
                            raise ValueError("取得量が想定サイズを超えました。")
                        stream.write(block)
                        if time.monotonic() - last_log >= 15:
                            print(f"取得中: {total / EXPECTED_BYTES:.1%}", flush=True)
                            last_log = time.monotonic()
        details = verify(partial)
        partial.replace(final)
    else:
        details = verify(final)
    record = {"status": "verified", "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "source_url": URL, "bytes": EXPECTED_BYTES, **details,
              "source_note_ja": "独立した公開アーカイブです。元の作成者の承認や原版との完全な同一性は証明していません。",
              "terms_ja": "JVS 作成者の音声利用条件に従い、個人・非商用研究用に扱います。データは配布しません。"}
    (directory / "取得記録.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print("取得済みデータの検証が完了しました。", flush=True)


if __name__ == "__main__":
    main()
