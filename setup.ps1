param(
    [ValidateSet('CPU', 'CUDA121')][string]$Device = 'CPU',
    [string]$PythonPath = '',
    [switch]$ModelOnly
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

try {
    if (-not $PythonPath) {
        $configPath = Join-Path $PSScriptRoot 'runtime.local.json'
        if (Test-Path -LiteralPath $configPath) {
            $configured = (Get-Content -LiteralPath $configPath -Raw -Encoding UTF8 | ConvertFrom-Json).python
            if ($configured -and (Test-Path -LiteralPath $configured)) { $PythonPath = $configured }
        }
    }
    if (-not $PythonPath) {
        $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
        if ($pythonCommand) { $PythonPath = $pythonCommand.Source }
    }
    if (-not $PythonPath) {
        $launcher = Get-Command py -ErrorAction SilentlyContinue
        if ($launcher) {
            foreach ($version in @('-3.12', '-3.11', '-3.10')) {
                try {
                    $resolved = & $launcher.Source $version -c 'import sys; print(sys.executable)' 2>$null
                    if ($LASTEXITCODE -eq 0) { $PythonPath = [string]@($resolved)[-1]; break }
                } catch { }
            }
        }
    }
    if (-not $PythonPath) { throw 'Python 3.10～3.12 をインストールしてから、もう一度実行してください。' }
    & $PythonPath -c 'import sys; assert (3,10) <= sys.version_info[:2] <= (3,12)'
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.10～3.12 を指定してください。例: setup.cmd -PythonPath C:\Python312\python.exe' }
    if ($ModelOnly) {
        & $PythonPath (Join-Path $PSScriptRoot 'models\download_model.py')
        exit $LASTEXITCODE
    }

    Write-Host 'このフォルダー内に必要なライブラリを準備します。'
    $runtimePython = $PythonPath
    $targetArguments = @()
    $venvPath = Join-Path $PSScriptRoot '.venv'
    $venvPython = Join-Path $venvPath 'Scripts\python.exe'
    $isBundled = $PythonPath -match 'codex-primary-runtime'
    if (-not $isBundled) {
        & $PythonPath -m venv $venvPath
        if ($LASTEXITCODE -ne 0) { throw '専用環境を作成できませんでした。Python のインストールを確認してください。' }
        $runtimePython = $venvPython
    } else {
        $targetPath = Join-Path $PSScriptRoot '.runtime'
        New-Item -ItemType Directory -Path $targetPath -Force | Out-Null
        $targetArguments = @('--target', $targetPath, '--upgrade')
    }
    $indexUrl = if ($Device -eq 'CUDA121') { 'https://download.pytorch.org/whl/cu121' } else { 'https://download.pytorch.org/whl/cpu' }
    Write-Host ('PyTorch を取得中: ' + $Device + ' （数分かかることがあります）')
    & $runtimePython -m pip install @targetArguments --index-url $indexUrl 'torch==2.5.1' 'torchaudio==2.5.1'
    if ($LASTEXITCODE -ne 0) { throw 'PyTorch のインストールに失敗しました。通信環境と空き容量を確認してください。' }
    & $runtimePython -m pip install @targetArguments 'numpy==2.2.6' 'soundfile==0.13.1'
    if ($LASTEXITCODE -ne 0) { throw '音声ライブラリのインストールに失敗しました。' }
    @{ python = $runtimePython } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $PSScriptRoot 'runtime.local.json') -Encoding UTF8
    & $runtimePython (Join-Path $PSScriptRoot 'models\download_model.py')
    if ($LASTEXITCODE -ne 0) { throw 'モデルを取得できませんでした。モデル取得.cmd で再試行できます。' }
    & (Join-Path $PSScriptRoot 'start.ps1') -CheckOnly
    if ($LASTEXITCODE -ne 0) { throw '起動確認に失敗しました。表示されたメッセージを確認してください。' }
    Write-Host '準備ができました。start.cmd をダブルクリックして起動してください。' -ForegroundColor Green
} catch {
    Write-Host ('準備を完了できませんでした。' + [Environment]::NewLine + $_.Exception.Message) -ForegroundColor Red
    exit 1
}
