$ErrorActionPreference='Stop'
$taskRoot=[System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$taskFdsRoot=Join-Path $taskRoot 'output\fds'
$taskStaging=[System.IO.Path]::GetFullPath((Join-Path $taskFdsRoot 'gaehwasan_20261102_gas'))
$taskActive=[System.IO.Path]::GetFullPath((Join-Path $taskFdsRoot 'gaehwasan'))
$taskArchive=[System.IO.Path]::GetFullPath((Join-Path $taskFdsRoot 'gaehwasan_before_20261102_gas'))
foreach ($taskPath in @($taskStaging,$taskActive,$taskArchive)) {
    if (-not $taskPath.StartsWith($taskFdsRoot+[System.IO.Path]::DirectorySeparatorChar,[System.StringComparison]::OrdinalIgnoreCase)) { throw 'FDS 활성화 경로가 지정된 폴더를 벗어났습니다.' }
    if ((Test-Path -LiteralPath $taskPath) -and ((Get-Item -LiteralPath $taskPath).Attributes -band [System.IO.FileAttributes]::ReparsePoint)) { throw 'FDS 활성화 폴더에 링크를 사용할 수 없습니다.' }
}
if (Test-Path -LiteralPath $taskArchive) { throw '기존 FDS 보관 폴더를 덮어쓸 수 없습니다.' }
$taskReport=Get-Content -LiteralPath (Join-Path $taskStaging 'gas_validation.json') -Raw -Encoding UTF8 | ConvertFrom-Json
if ($taskReport.status -ne 'validated' -or $taskReport.validated_connector_breathing_positions -lt 1) { throw '가스 농도장과 통로 호흡 위치 검증이 완료되지 않았습니다.' }
$taskChecks=@(
    @{Path=(Join-Path $taskRoot 'configs\scenarios\gaehwasan.json');Hash=$taskReport.source_config_sha256},
    @{Path=(Join-Path $taskStaging 'gaehwasan.fds');Hash=$taskReport.deck_sha256},
    @{Path=(Join-Path $taskStaging 'derived\fields.npz');Hash=$taskReport.fields_sha256}
)
foreach ($taskCheck in $taskChecks) {
    if ((Get-FileHash -LiteralPath $taskCheck.Path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $taskCheck.Hash) { throw '가스 검증 후 입력 또는 계산 파일이 변경되었습니다.' }
}
$taskDeck=Join-Path $taskRoot 'configs\fds\gaehwasan.fds'
$taskTempDeck=Join-Path $taskRoot 'configs\fds\gaehwasan.gas.tmp'
$taskPreviousDeck=[System.IO.File]::ReadAllBytes($taskDeck)
Copy-Item -LiteralPath (Join-Path $taskStaging 'gaehwasan.fds') -Destination $taskTempDeck
if (Test-Path -LiteralPath $taskActive) { Move-Item -LiteralPath $taskActive -Destination $taskArchive }
try {
    Move-Item -LiteralPath $taskStaging -Destination $taskActive
    Move-Item -LiteralPath $taskTempDeck -Destination $taskDeck -Force
} catch {
    if ((Test-Path -LiteralPath $taskActive) -and -not (Test-Path -LiteralPath $taskStaging)) { Move-Item -LiteralPath $taskActive -Destination $taskStaging }
    if ((Test-Path -LiteralPath $taskArchive) -and -not (Test-Path -LiteralPath $taskActive)) { Move-Item -LiteralPath $taskArchive -Destination $taskActive }
    [System.IO.File]::WriteAllBytes($taskDeck,$taskPreviousDeck)
    throw
}
$taskReaderCache=Join-Path $taskActive 'gaehwasan.pickle'
if (Test-Path -LiteralPath $taskReaderCache) { Move-Item -LiteralPath $taskReaderCache -Destination (Join-Path $taskActive 'reader_before_activation.pickle') }
$taskReport.activated=$true
[System.IO.File]::WriteAllText((Join-Path $taskActive 'gas_validation.json'),($taskReport | ConvertTo-Json -Depth 8),[System.Text.UTF8Encoding]::new($false))
Write-Output '실제 FDS 가스 농도장을 연결했습니다. 이전 FDS와 승객 결과는 보존했습니다.'
