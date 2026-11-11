# AntiFake JP

日本語の操作画面で録音に保護用の摂動を加える研究用ツールと、日本語データによる話者エンコーダーの追加学習パイプラインです。音声の処理はローカルで行います。

AntiFake の考え方を参考にした、RTVC 単一エンコーダーによる独自の簡略実装です。原論文のアンサンブル、参考話者選択、知覚損失を完全再現したものではありません。クローン防御効果を保証する製品ではありません。

**実行に必要な原版モデル・日本語追加学習モデルの重みと登録設定を同梱しています。保護処理を使うために学習し直す必要はありません。** 学習・評価レポート、スコア、録音、生成音声、データセット、実行ログは含めません。

## すぐに使う

1. GitHub の「Code」→「Download ZIP」で取得し、フォルダー全体を展開します。`git clone https://github.com/fnnns/antifake-jp.git` でも取得できます。
2. Python 3.10～3.12 を用意し、`setup.cmd` を実行して必要なライブラリをインストールします。セットアップは同梱モデルの検証も行います。
3. `start.cmd` を実行し、「使用するモデル」で「日本語で微調整したモデル」を選びます。録音と保存先を選んで処理を開始してください。

Python とライブラリの初回準備にはインターネット接続が必要です。準備後の録音処理はローカルで行い、学習データの取得は不要です。重みの説明と SHA-256 は [同梱モデル](docs/MODELS.md) にあります。

## 機能

- 日本語のデスクトップ画面とコマンドライン。
- WAV / FLAC / OGG の読み込み、処理の進捗表示、中止、WAV の試聴。
- 16 kHz・モノラル・16 bit PCM WAV の出力。
- 元の録音・既存の出力の上書き防止。
- 元のモデルと、自分で追加学習した日本語モデルの切り替え。
- JVS 形式 ZIP の直接前処理、話者単位の分割、GE2E による追加学習。
- 検証集合によるモデル選択、別のテスト集合による評価、重みの検証と登録。

## 必要な環境

- Windows 10 / 11。
- Python 3.10～3.12（Tkinter を含む）。
- CPU、または CUDA 12.1 版 PyTorch に対応する NVIDIA GPU とドライバー。
- 初回のライブラリ・元モデル・学習データの取得に使うインターネット接続。

通常の保護処理は CPU でも実行できます。追加学習には NVIDIA GPU を推奨します。データと生成する特徴量を保存できる空き容量を用意してください。

## セットアップ

このフォルダーで、CPU 版を準備する場合：

```powershell
.\setup.cmd
```

NVIDIA GPU 版を準備する場合：

```powershell
.\setup.cmd -Device CUDA121
```

Python の場所を指定する場合：

```powershell
.\setup.cmd -Device CUDA121 -PythonPath "C:\Python312\python.exe"
```

専用環境 `.venv` を作成し、必要なライブラリを取得します。同梱した原版・日本語モデルは SHA-256 と読込を確認します。原版がない場合だけ、公開されている元の RTVC モデルを取得します。システム全体の Python ライブラリは変更しません。

## 録音を処理する

`start.cmd` をダブルクリックします。

1. 「保護する録音」を選びます。最初は一人で話した短い日本語録音が適しています。
2. 出力 WAV の保存先を確認します。
3. 「使用するモデル」で「日本語で微調整したモデル」を選びます。「原版の事前学習モデル」と比較することもできます。
4. 必要に応じて反復回数・SNR 下限・デバイスを設定します。
5. 「録音の保護を開始」を押し、処理後に聞こえ方を確認します。

日本語モデルは登録済みです。「自動」は原版モデルを使う既存の設定を維持しています。日本語モデルを使う場合は、画面または `--model japanese` で明示的に選んでください。

コマンドラインの例：

```powershell
.\.venv\Scripts\python.exe cli.py input.wav output.wav --model japanese --device auto --iterations 150 --snr 30
```

処理できる長さは 0.5 秒～5 分です。出力は 16 kHz・モノラルになるため、元の高音域・ステレオ情報を含むファイルは保存しておいてください。SNR は波形の変化量を表す指標で、聞きやすさや防御効果を直接保証しません。

## 日本語モデルを追加学習する

ここからは、自分で追加学習し直す場合だけの手順です。同梱モデルで録音を処理するだけなら不要です。GPU 版セットアップ後に `学習を実行.cmd` を実行できます。データがない場合は取得し、前処理、追加学習、推論用モデルの登録を順に行います。学習データの取得元と利用条件を、先に [データについて](docs/DATA.md) で確認してください。

手動で実行する例：

```powershell
.\.venv\Scripts\python.exe training/pipeline.py preprocess --archive "C:\datasets\jvs_ver1.zip" --out training/data
.\.venv\Scripts\python.exe training/pipeline.py train --data training/data --pretrained models/encoder.pt --run-dir training/runs/my_run --steps 3000 --eval-every 500 --speakers 8 --utterances 5 --lr 0.00001 --device cuda
.\.venv\Scripts\python.exe training/register_model.py --run-dir training/runs/my_run --output-name encoder_ja_custom.pt
```

追加学習は、元の重みを読み込んで新しい Adam を作り、同じ話者の表現を近づけ、異なる話者を区別する GE2E 損失で行います。録音の話者 ID が必要で、文字起こしは学習入力に使いません。前処理は通常の読み上げを対象とし、ささやき声と裏声を除きます。

詳しくは [追加学習の説明](training/README.md) を参照してください。モデル登録の初期値では「自動」は元のモデルを使います。自分の日本語モデルを自動選択にする場合は、登録時に `--activate` を指定します。

## ソースの構成

```text
app.py                       日本語の操作画面
cli.py                       コマンドライン
core.py                      推論構造・摂動の最適化・音声入出力
models/download_model.py     元のモデルの取得と検証
models/verify_models.py      同梱した両モデルの検証
models/encoder.pt            原版の事前学習済み重み
models/encoder_ja.pt         日本語で追加学習した推論用重み
models/registry.json         同梱モデルの登録設定
models/registry.example.json 登録情報のひな形
training/pipeline.py         前処理・GE2E 追加学習・検証・モデル選択
training/register_model.py   推論用モデルの検証と登録
training/train.ps1           一連の追加学習の実行
tests/                       モデル登録情報の確認
docs/                        データと公開対象の説明
third_party/                 参照元の許諾文
```

## 確認方法

セットアップ後に、モデル登録情報の確認を実行できます。

```powershell
.\.venv\Scripts\python.exe models/verify_models.py
.\.venv\Scripts\python.exe tests/test_model_registry.py
```

GPU 環境では、小規模な合成データで追加学習の流れを確認できます。

```powershell
.\.venv\Scripts\python.exe training/test_pipeline.py
```

これらは実装の動作確認です。日本語のクローン防御性能を示すものではありません。

## 公開対象と生成物

実行用の固定された 2 個のモデルと登録設定を公開しています。自分で学習や処理を行って作られる追加の重み、スコア、レポート、音声は各自の環境へ保存され、`.gitignore` で除外します。公開対象の方針は [公開するファイル](docs/PUBLISHING.md) にあります。

日本語モデルには推論に使うテンソルだけを収録し、学習履歴・最適化状態・評価スコアを取り除いています。過去の学習・評価レポートは公開しません。

## 制限

- 単一の声紋モデルを用いる研究用実装です。
- 最適化に使ったモデルの声紋変化を、そのまま他のクローンモデルの失敗率として扱えません。
- 話者認証の EER の改善も、クローン防御効果の証明ではありません。
- 圧縮、再録音、雑音除去、別の生成モデルによって、保護効果が変わる可能性があります。
- 本人が利用する権利を持つ録音と、条件を確認した学習データを使用してください。

## 出典

- [AntiFake](https://github.com/WUSTL-CSPL/AntiFake) / [原論文](https://doi.org/10.1145/3576915.3623209)
- [Real-Time Voice Cloning](https://github.com/CorentinJ/Real-Time-Voice-Cloning)
- [JVS コーパス](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus)

第三者の成果と許諾文については [THIRD_PARTY.md](THIRD_PARTY.md) を参照してください。
