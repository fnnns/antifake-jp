"""同梱したモデルの検証値と読込を確認する。通信や音声の取得は行わない。"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import bootstrap  # noqa: E402, F401
import torch  # noqa: E402
from core import _load_encoder, get_model_spec  # noqa: E402


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    try:
        for name in ("original", "japanese"):
            spec = get_model_spec(name)
            model, _ = _load_encoder(torch.device("cpu"), spec)
            if not all(torch.isfinite(value).all() for value in model.state_dict().values()):
                raise ValueError("モデルに無効な数値があります。")
            print(f"確認済み：{spec['display_name_ja']}（{spec['path'].name}）")
        print("両モデルを利用できます。日本語版は画面のモデル一覧から選択してください。")
        return 0
    except Exception as error:
        print(f"モデルを確認できません：{error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
