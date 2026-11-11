"""学習結果を検証し、推論用の日本語モデルをアプリへ登録します。"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import bootstrap  # noqa: E402, F401
import torch  # noqa: E402
from core import MODEL_SHA256, SpeakerEncoder  # noqa: E402


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="追加学習したモデルを検証し、アプリへ登録します。")
    parser._optionals.title = "オプション"
    parser.add_argument("--run-dir", type=Path, required=True, help="学習結果があるフォルダー")
    parser.add_argument("--output-name", default="encoder_ja.pt", help="推論用モデルの新しいファイル名")
    parser.add_argument("--activate", action="store_true", help="自動選択で日本語モデルを使います。")
    args = parser.parse_args()
    run = args.run_dir.resolve()
    report_path = run / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    checkpoint_path = run / "best.pt"
    if report["status"] != "completed" or report["selected_step"] < 1 or report["completed_steps"] < report["selected_step"]:
        raise ValueError("完了した追加学習の記録がありません。")
    if report["initial_checkpoint_sha256"] != MODEL_SHA256 or sha256(ROOT / "models" / "encoder.pt") != MODEL_SHA256:
        raise ValueError("元のモデルの検証値が一致しません。")
    if sha256(checkpoint_path) != report["checkpoint_sha256"]:
        raise ValueError("学習結果とチェックポイントの検証値が一致しません。")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["step"] != report["selected_step"] or checkpoint["data_manifest_sha256"] != report["data_manifest_sha256"]:
        raise ValueError("選択された反復またはデータ記録が一致しません。")
    state = {name: value.detach().cpu().clone() for name, value in checkpoint["model_state"].items()}
    if not all(torch.isfinite(value).all() for value in state.values()):
        raise ValueError("モデルの重みに無効な数値があります。")
    original = torch.load(ROOT / "models" / "encoder.pt", map_location="cpu", weights_only=True)["model_state"]
    expected = SpeakerEncoder().state_dict()
    inference = SpeakerEncoder()
    inference.load_state_dict({name: state[name] for name in expected}, strict=True)
    changed = [name for name in expected if not torch.equal(state[name], original[name])]
    if not changed:
        raise ValueError("追加学習による推論用重みの更新が確認できません。")
    model_dir = (ROOT / "models").resolve()
    output = (model_dir / args.output_name).resolve()
    if output.parent != model_dir or output.suffix != ".pt" or output.name == "encoder.pt":
        raise ValueError("推論モデルには models 内の新しい .pt ファイル名を指定してください。")
    if output.exists():
        raise ValueError("同名のモデルがあります。別のファイル名を指定してください。")
    metadata = {
        "language": "ja", "selected_step": report["selected_step"],
        "completed_steps": report["completed_steps"],
        "initial_checkpoint_sha256": MODEL_SHA256,
        "data_manifest_sha256": report["data_manifest_sha256"],
        "training_checkpoint_sha256": report["checkpoint_sha256"],
        "training_report_sha256": sha256(report_path),
        "training_code_sha256": report["training_code_sha256"],
        "baseline_test_eer": report["baseline_test"]["eer"],
        "selected_test_eer": report["selected_test"]["eer"],
        "note_ja": "JVS の通常日本語音声で追加学習した話者モデルです。クローン防御率は未評価です。",
    }
    with output.open("xb") as stream:
        torch.save({"model_state": state, "metadata": metadata}, stream)
    # 推論用に保存し直したファイルを独立して読み込み、検証します。
    restored = torch.load(output, map_location="cpu", weights_only=True)
    for name in expected:
        if not torch.equal(restored["model_state"][name], state[name]):
            raise ValueError("推論用モデルの保存確認に失敗しました。")
    digest = sha256(output)
    registry_path = model_dir / "registry.json"
    if registry_path.is_file():
        registry = json.loads(registry_path.read_text(encoding="utf-8-sig"))
    else:
        # 公開ソースには実際の登録情報を含めず、初回の登録時に作成します。
        registry = {"schema_version": 1, "active": "original", "models": {
            "original": {"path": "encoder.pt", "sha256": MODEL_SHA256,
                         "display_name_ja": "原版の事前学習モデル"}}}
    registry["models"]["japanese"] = {"path": output.name, "sha256": digest,
        "display_name_ja": "日本語で微調整したモデル（JVS）",
        "training_report": report_path.relative_to(ROOT).as_posix()}
    if args.activate:
        registry["active"] = "japanese"
    temporary = registry_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(registry_path)
    record = {"status": "registered", "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "model_path": output.relative_to(ROOT).as_posix(), "model_sha256": digest,
              "updated_inference_parameters": changed, "active_model": registry["active"], **metadata}
    (model_dir / "日本語モデル登録記録.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print("日本語モデルの重み更新、保存、検証、登録が完了しました。", flush=True)
    print(json.dumps({"model_sha256": digest, "active_model": registry["active"], "selected_step": report["selected_step"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
