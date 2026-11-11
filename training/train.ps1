param(
    [ValidateRange(1, 100000)][int]$Steps = 3000,
    [ValidateSet('cuda', 'cpu')][string]$Device = 'cuda'
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot

try {
    $pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $pythonPath)) {
        $localConfig = Join-Path $projectRoot 'runtime.local.json'
        if (-not (Test-Path -LiteralPath $localConfig)) {
            throw 'setup.cmd を実行し、専用の実行環境を用意してください。'
        }
        $pythonPath = (Get-Content -LiteralPath $localConfig -Raw -Encoding UTF8 | ConvertFrom-Json).python
    }
    if (-not $pythonPath -or -not (Test-Path -LiteralPath $pythonPath)) {
        throw 'Python の場所を確認できません。setup.cmd を実行してください。'
    }
    Write-Host '日本語データを検証します。データがない場合は約 3.5 GB を取得します。'
    & $pythonPath -X utf8 'training\download_data.py'
    if ($LASTEXITCODE -ne 0) { throw '学習データの取得・検証に失敗しました。' }
    & $pythonPath -X utf8 'training\pipeline.py' preprocess --archive 'training\raw\jvs_ver1.zip' --out 'training\data'
    if ($LASTEXITCODE -ne 0) { throw '音声の前処理に失敗しました。' }
    $stamp = [DateTimeOffset]::UtcNow.ToOffset([TimeSpan]::FromHours(9)).ToString('yyyyMMdd_HHmmss')
    $runName = 'jvs_ja_' + $stamp
    $runPath = Join-Path 'training\runs' $runName
    & $pythonPath -X utf8 'training\pipeline.py' train --data 'training\data' --pretrained 'models\encoder.pt' --run-dir $runPath --steps $Steps --eval-every 500 --patience 3 --min-steps 1000 --speakers 8 --utterances 5 --lr 0.00001 --device $Device
    if ($LASTEXITCODE -ne 0) { throw '追加学習を完了できませんでした。' }
    & $pythonPath -X utf8 'training\register_model.py' --run-dir $runPath --output-name ('encoder_' + $runName + '.pt')
    if ($LASTEXITCODE -ne 0) { throw '学習は終了しましたが、アプリへの登録に失敗しました。学習結果は保持しています。' }
    Write-Host '追加学習とモデルの登録が完了しました。学習結果.md を確認してください。' -ForegroundColor Green
    Write-Host ('保存先: ' + $runPath)
} catch {
    Write-Host $_.Exception.Message -ForegroundColor Red
    exit 1
}
