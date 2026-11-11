param([switch]$CheckOnly)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

function Show-LaunchError([string]$Message) {
    if ($CheckOnly) { Write-Error $Message; return }
    Add-Type -AssemblyName System.Windows.Forms
    [System.Windows.Forms.MessageBox]::Show($Message, 'AntiFake JP', 'OK', 'Error') | Out-Null
}

try {
    $candidates = [System.Collections.Generic.List[string]]::new()
    $venvPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
    if (Test-Path -LiteralPath $venvPython) { $candidates.Add($venvPython) }
    $configPath = Join-Path $PSScriptRoot 'runtime.local.json'
    if (Test-Path -LiteralPath $configPath) {
        try {
            $configured = (Get-Content -LiteralPath $configPath -Raw -Encoding UTF8 | ConvertFrom-Json).python
            if ($configured -and (Test-Path -LiteralPath $configured)) { $candidates.Add($configured) }
        } catch { }
    }
    foreach ($commandName in @('python', 'python3')) {
        $command = Get-Command $commandName -ErrorAction SilentlyContinue
        if ($command) { $candidates.Add($command.Source) }
    }
    $launcher = Get-Command py -ErrorAction SilentlyContinue
    if ($launcher) {
        foreach ($version in @('-3.12', '-3.11', '-3.10')) {
            try {
                $resolved = & $launcher.Source $version -c 'import sys; print(sys.executable)' 2>$null
                if ($LASTEXITCODE -eq 0 -and $resolved -and (Test-Path -LiteralPath @($resolved)[-1])) {
                    $candidates.Add([string]@($resolved)[-1])
                }
            } catch { }
        }
    }
    $selected = $null
    $lastDetail = ''
    foreach ($candidate in ($candidates | Select-Object -Unique)) {
        try {
            $ErrorActionPreference = 'Continue'
            $check = & $candidate -c 'import sys; assert (3,10) <= sys.version_info[:2] <= (3,12); import bootstrap; import tkinter, torch, torchaudio, numpy, soundfile; print(1)' 2>&1
            $checkExitCode = $LASTEXITCODE
            $ErrorActionPreference = 'Stop'
            if ($checkExitCode -eq 0) { $selected = $candidate; break }
            $lastDetail = ($check | Out-String).Trim()
        } catch { $ErrorActionPreference = 'Stop'; $lastDetail = $_.Exception.Message }
    }
    if (-not $selected) {
        throw "必要な Python とライブラリを確認できません。setup.cmd を実行してください。`nPython 3.10～3.12 に対応しています。`n`n$lastDetail"
    }
    if ($CheckOnly) {
        Write-Output "使用する Python: $selected"
        Write-Output '必要なライブラリ: 確認済み'
        exit 0
    }
    $windowedPython = Join-Path (Split-Path -Parent $selected) 'pythonw.exe'
    if (-not (Test-Path -LiteralPath $windowedPython)) { $windowedPython = $selected }
    $appPath = Join-Path $PSScriptRoot 'app.py'
    $launchArguments = '"{0}"' -f $appPath
    Start-Process -FilePath $windowedPython -ArgumentList $launchArguments -WorkingDirectory $PSScriptRoot -WindowStyle Hidden
} catch {
    Show-LaunchError ("起動できませんでした。`n`n" + $_.Exception.Message)
    exit 1
}
