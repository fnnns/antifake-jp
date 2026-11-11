"""日本語で案内するコマンドラインの起動処理。"""
import argparse
import json
from pathlib import Path
import sys
import bootstrap  # noqa: F401


class JapaneseHelpFormatter(argparse.HelpFormatter):
    """引数の一覧に表示する見出しを日本語にする。"""

    def _format_usage(self, usage, actions, groups, prefix):
        return super()._format_usage(usage, actions, groups, prefix or "使い方: ")


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(
        description="AntiFake-JP 実験版: 日本語録音に保護用の摂動を加えます。",
        formatter_class=JapaneseHelpFormatter, add_help=False)
    parser._positionals.title = "必須の引数"
    parser._optionals.title = "オプション"
    parser.add_argument("-h", "--help", action="help", help="この使い方を表示して終了します")
    parser.add_argument("input", type=Path, help="元の録音 WAV / FLAC / OGG")
    parser.add_argument("output", type=Path, help="保存先 WAV（既存ファイルは上書きしません）")
    parser.add_argument("--target", type=Path, help="任意: 別の話者の参照音声")
    parser.add_argument("--iterations", type=int, default=150, help="繰り返し回数（初期値: 150）")
    parser.add_argument("--snr", type=float, default=30, help="最小 SNR、単位 dB（初期値: 30）")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="処理するデバイス（初期値: auto、GPU を優先）")
    parser.add_argument("--model", choices=["auto", "original", "japanese"], default="auto", help="使用するモデル: 自動、原版、日本語微調整版（初期値: auto）")
    args = parser.parse_args()
    from core import protect_file, ProtectionConfig
    try:
        result = protect_file(args.input, args.output, args.target,
                              ProtectionConfig(iterations=args.iterations, min_snr_db=args.snr, device=args.device, model_name=args.model),
                              progress=lambda done, total, msg: print(f"[{done}/{total}] {msg}", flush=True))
        print(json.dumps({key: result[key] for key in ["status", "output_path", "report_path", "snr_db", "source_cosine_before_after", "elapsed_seconds"]}, ensure_ascii=False, indent=2))
        return 0
    except KeyboardInterrupt:
        print("中止しました。", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"エラー: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
