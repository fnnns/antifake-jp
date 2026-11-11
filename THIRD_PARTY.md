# 出典と第三者の成果

## AntiFake

- 論文：Zhiyuan Yu、Shixuan Zhai、Ning Zhang。*AntiFake: Using Adversarial Audio to Prevent Unauthorized Speech Synthesis*、CCS 2023。
- [公式リポジトリ](https://github.com/WUSTL-CSPL/AntiFake)
- [論文](https://doi.org/10.1145/3576915.3623209)

本版は、公開された声紋への摂動という考え方を参考に、最適化、録音の入出力、日本語画面、追加学習手順を独自に簡略実装したものです。AntiFake の `run.py`、TTS / AVC / Tortoise の構成、話者データベース、デモ音声は収録していません。元のアンサンブルや AHP による参考話者選択を再現した版ではありません。

## Real-Time Voice Cloning

- [作者のリポジトリ](https://github.com/CorentinJ/Real-Time-Voice-Cloning)
- [元のモデルの公開ページ](https://huggingface.co/CorentinJ/SV2TTS)

`core.py` の話者エンコーダーの推論構造は、MIT ライセンスの RTVC を参考にしています。元の著作権表示と許諾文は [third_party/RTVC-LICENSE.txt](third_party/RTVC-LICENSE.txt) に改変せず保持しています。

構造は 40 Mel チャンネル、各層 256 ユニットの 3 層 LSTM、256 次元の出力、ReLU、L2 正規化です。公開された元の重みは `models/download_model.py` から取得できます。元の重みと、自分で追加学習した重みは分けて管理してください。

このリポジトリには元の重みも追加学習した重みも含めません。第三者に由来する個々の条件を、本版全体の一律のライセンス表示に置き換えるものではありません。

## JVS

- [JVS 作成者の公開ページ](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus)
- [JVS 論文](https://arxiv.org/abs/1908.06248)

音声、タグ、テキストにはそれぞれの利用条件があります。詳しくは [docs/DATA.md](docs/DATA.md) と作成者のページを参照してください。録音・テキスト・特徴量・取得済みデータは収録していません。

## ライブラリ

PyTorch、torchaudio、NumPy、SoundFile、Tkinter などを使用します。ライブラリ自体はこのリポジトリへコピーせず、セットアップ時に取得します。各ライブラリの許諾文は、それぞれの配布物に従ってください。
