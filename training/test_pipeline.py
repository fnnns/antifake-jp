"""合成した小規模データで前処理から保存までの結合動作を確認する。"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import tempfile
import zipfile

import pipeline
import numpy as np
import soundfile as sf
import torch


def main():
    folder = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(prefix="学習結合試験_", dir=folder) as temporary:
        temporary = Path(temporary)
        archive = temporary / "小規模試験.zip"
        sample_rate = 24000
        times = np.arange(sample_rate * 2, dtype=np.float32) / sample_rate
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as package:
            metadata = []
            for speaker in range(1, 21):
                identity = f"jvs{speaker:03d}"
                metadata.append(f"{identity}\t{'M' if speaker <= 10 else 'F'}\t100\t200")
                for utterance in range(10):
                    rng = np.random.default_rng(speaker * 100 + utterance)
                    wave = (0.08 * np.sin(2 * np.pi * (100 + speaker * 9) * times)
                            + 0.04 * np.sin(2 * np.pi * (210 + speaker * 17 + utterance) * times)
                            + rng.normal(0, 0.003, len(times))).astype(np.float32)
                    stream = io.BytesIO()
                    sf.write(stream, wave, sample_rate, format="WAV", subtype="PCM_16")
                    package.writestr(f"jvs_ver1/{identity}/parallel100/wav24kHz16bit/{utterance:03d}.wav", stream.getvalue())
            package.writestr("jvs_ver1/gender_f0range.txt", "\n".join(metadata))
        data = temporary / "特徴量"
        manifest = pipeline.preprocess(argparse.Namespace(archive=str(archive), out=str(data), seed=pipeline.SEED))
        assert manifest["gender_stratified"]
        assert [len(manifest["speaker_splits"][key]) for key in ("train", "val", "test")] == [16, 2, 2]
        run = temporary / "学習結果"
        report = pipeline.train(argparse.Namespace(
            command="train", data=str(data), pretrained=str(pipeline.ROOT / "models" / "encoder.pt"),
            run_dir=str(run), steps=10, eval_every=5, log_every=5, patience=3, min_steps=10,
            speakers=8, utterances=5, lr=1e-5, seed=pipeline.SEED, device="cuda",
        ))
        assert report["completed_steps"] == 10 and report["selected_step"] >= 1
        assert report["training_speakers"] == 16 and report["validation_speakers"] == 2 and report["test_speakers"] == 2
        trained = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
        initial = torch.load(pipeline.ROOT / "models" / "encoder.pt", map_location="cpu", weights_only=True)
        assert trained["step"] >= 1
        assert not torch.equal(trained["model_state"]["linear.weight"], initial["model_state"]["linear.weight"])
        assert report["checkpoint_sha256"] != report["initial_checkpoint_sha256"]
        result = {"status": "passed", "scope_ja": "合成データによる実装検証。日本語の学習結果ではありません。",
                  "checks_ja": ["JVS 形式 ZIP の直接読み込み", "性別を考慮した話者分割", "学習・検証・テストの独立性",
                                "全体の追加学習と重み更新", "実際の学習後モデルの選択と保存", "安全な重み再読み込み"],
                  "synthetic_speakers": 20, "synthetic_utterances": manifest["utterances"],
                  "completed_steps": report["completed_steps"], "selected_step": report["selected_step"]}
    result["temporary_data_removed"] = not temporary.exists()
    pipeline.write_json(folder / "integration_test.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
