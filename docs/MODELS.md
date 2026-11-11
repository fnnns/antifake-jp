# 同梱モデル

録音の保護を実行するための重みを `models/` に同梱しています。追加学習や JVS の取得は、通常の保護処理には必要ありません。

| ファイル | 内容 | サイズ（バイト） |
| --- | --- | ---: |
| `encoder.pt` | 公開されている原版の RTVC 話者エンコーダー | 17,090,379 |
| `encoder_ja.pt` | JVS の日本語読み上げ音声で追加学習した推論用エンコーダー | 5,699,234 |
| `registry.json` | 両モデルの保存場所・検証値・選択設定 | — |

## 検証値

SHA-256 は次のとおりです。アプリは読み込み前に検証します。

```text
encoder.pt
39373b86598fa3da9fcddee6142382efe09777e8d37dc9c0561f41f0070f134e

encoder_ja.pt
37e043b28f6d3d0a901f67b14171e2f5bfbba11f7283f263c937ab901d6c9b91
```

日本語版は `model_state` のうち推論に必要なテンソルだけを保存しています。元の追加学習モデルと推論用パラメーターは同一です。学習履歴、評価スコア、最適化状態、音声、個別の発話の情報を含めません。原版は、公開されているチェックポイントを改変せず保持しています。

セットアップは両モデルの検証と読み込み確認を実行します。手動でも確認できます。

```powershell
.\.venv\Scripts\python.exe models/verify_models.py
```

## 選び方

日本語版は画面の「日本語で微調整したモデル」、または `--model japanese` で使えます。「自動」は既存の原版設定を維持しています。モデルを選べることや追加学習の完了は、音声クローン防御性能を保証しません。

## 出典と利用条件

- 原版： [Real-Time Voice Cloning](https://github.com/CorentinJ/Real-Time-Voice-Cloning)。元ファイルは [AntiFake の固定版](https://github.com/WUSTL-CSPL/AntiFake/blob/0eb55a12589ccf8480acef67ea2beec46bea15f3/saved_models/default/encoder.pt) から取得したものです。RTVC の許諾文は [RTVC-LICENSE.txt](../third_party/RTVC-LICENSE.txt) にあります。
- 日本語追加学習のデータ： [JVS コーパスの公開元と利用条件](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus)。研究・個人利用を想定しています。商用目的で利用する場合は公開元の条件を確認してください。学習済みモデルへの個別の許諾条件は、データの公開ページでは明示されていません。
- データセットそのものは同梱しません。詳細は [データについて](DATA.md) と [出典と第三者の成果](../THIRD_PARTY.md) を参照してください。
