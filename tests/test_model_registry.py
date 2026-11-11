"""モデルの選択、原版の保持、登録情報の検証を音声処理なしで確認する。

登録ファイルは一時フォルダーへ作り、参照先だけを試験中に差し替える。
実際の登録ファイルとモデルの重みは書き換えない。
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import bootstrap  # noqa: E402, F401
import core  # noqa: E402


class ModelRegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_hash = core._sha256(core.MODEL_PATH)

    @classmethod
    def tearDownClass(cls):
        if core._sha256(core.MODEL_PATH) != cls.original_hash:
            raise AssertionError("試験中に元のモデルが変更されました。")

    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory(prefix="モデル登録確認_", dir=Path(__file__).parent)
        self.addCleanup(self.scratch.cleanup)
        self.registry_path = Path(self.scratch.name) / "registry.json"

    def registry(self, *, active="original", original=None, japanese=None):
        entries = {}
        if original is not None:
            entries["original"] = original
        if japanese is not None:
            entries["japanese"] = japanese
        self.registry_path.write_text(
            json.dumps({"schema_version": 1, "active": active, "models": entries}, ensure_ascii=False),
            encoding="utf-8")
        return patch.object(core, "REGISTRY_PATH", self.registry_path)

    def test_original_path_and_checksum_match_pinned_model(self):
        spec = core.get_model_spec("original")
        self.assertEqual(spec["name"], "original")
        self.assertEqual(spec["path"], core.MODEL_PATH.resolve())
        self.assertEqual(spec["sha256"], core.MODEL_SHA256)
        self.assertEqual(self.original_hash, core.MODEL_SHA256)

    def test_no_registry_uses_original(self):
        with patch.object(core, "REGISTRY_PATH", self.registry_path):
            self.assertFalse(self.registry_path.exists())
            self.assertEqual(core.get_model_spec("auto")["name"], "original")

    def test_missing_japanese_entry_has_japanese_guidance(self):
        with self.registry():
            with self.assertRaisesRegex(core.ProtectionError, "日本語モデルの学習・登録"):
                core.get_model_spec("japanese")

    def test_original_checksum_cannot_be_replaced(self):
        entry = {"path": "encoder.pt", "sha256": "0" * 64}
        with self.registry(original=entry):
            with self.assertRaisesRegex(core.ProtectionError, "保存場所または検証値"):
                core.get_model_spec("original")

    def test_original_path_cannot_be_replaced(self):
        entry = {"path": "別のモデル.pt", "sha256": core.MODEL_SHA256}
        with self.registry(original=entry):
            with self.assertRaisesRegex(core.ProtectionError, "保存場所または検証値"):
                core.get_model_spec("original")

    def test_invalid_checksum_format_is_rejected(self):
        entry = {"path": "encoder.pt", "sha256": "検証値として無効"}
        with self.registry(japanese=entry):
            with self.assertRaisesRegex(core.ProtectionError, "保存場所または検証値"):
                core.get_model_spec("japanese")

    def test_path_outside_model_directory_is_rejected(self):
        entry = {"path": "../外側のファイル.pt", "sha256": core.MODEL_SHA256}
        with self.registry(japanese=entry):
            with self.assertRaisesRegex(core.ProtectionError, "保存場所または検証値"):
                core.get_model_spec("japanese")

    def test_wrong_checksum_is_rejected_before_loading(self):
        entry = {"path": "encoder.pt", "sha256": "0" * 64}
        with self.registry(japanese=entry):
            spec = core.get_model_spec("japanese")
            with patch.object(core.torch, "load", side_effect=AssertionError("検証前に読み込んでいます。")) as load:
                with self.assertRaisesRegex(core.ProtectionError, "モデルの検証に失敗"):
                    core._load_encoder(core.torch.device("cpu"), spec)
                load.assert_not_called()

    def test_explicit_original_is_preserved_when_japanese_is_active(self):
        # 選択だけの仮登録として元の重みを指す。日本語の学習済みモデルとは扱わない。
        entry = {"path": "encoder.pt", "sha256": core.MODEL_SHA256,
                 "display_name_ja": "日本語モデル選択の確認用仮登録"}
        with self.registry(active="japanese", japanese=entry):
            self.assertEqual(core.get_model_spec("auto")["name"], "japanese")
            original = core.get_model_spec("original")
            self.assertEqual(original["name"], "original")
            self.assertEqual(original["path"], core.MODEL_PATH.resolve())
            self.assertEqual(original["sha256"], core.MODEL_SHA256)


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    unittest.main(verbosity=2)
