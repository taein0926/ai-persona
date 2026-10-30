$ErrorActionPreference='Stop'
$taskRoot=[System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$taskOutputRoot=Join-Path $taskRoot 'output/fds'
$taskStaging=[System.IO.Path]::GetFullPath((Join-Path $taskOutputRoot 'gaehwasan_20261029_layout12'))
$taskCurrent=[System.IO.Path]::GetFullPath((Join-Path $taskOutputRoot 'gaehwasan'))
$taskArchive=[System.IO.Path]::GetFullPath((Join-Path $taskOutputRoot 'gaehwasan_before_20261029_layout12'))
foreach ($taskPath in @($taskStaging,$taskCurrent,$taskArchive)) {
    if (-not $taskPath.StartsWith($taskOutputRoot+[System.IO.Path]::DirectorySeparatorChar,[System.StringComparison]::OrdinalIgnoreCase)) {
        throw 'FDS 작업 경로가 지정된 출력 폴더를 벗어났습니다.'
    }
}
$taskValidationPath=Join-Path $taskRoot 'output/qa/station_fire_20261029/validation.json'
$taskValidation=Get-Content -LiteralPath $taskValidationPath -Raw -Encoding UTF8 | ConvertFrom-Json
$taskPost=Get-Content -LiteralPath (Join-Path $taskStaging 'postprocess_status.json') -Raw -Encoding UTF8 | ConvertFrom-Json
if ($taskValidation.status -ne 'validated' -or $taskPost.phase -ne 'validated_ready_for_activation') { throw '새 FDS 결과 검증이 아직 완료되지 않았습니다.' }
if (Test-Path -LiteralPath $taskArchive) { throw '기존 보관 폴더를 덮어쓸 수 없습니다.' }
$taskConfigHash=(Get-FileHash -LiteralPath (Join-Path $taskRoot 'configs/scenarios/gaehwasan.json') -Algorithm SHA256).Hash.ToLowerInvariant()
if ($taskConfigHash -ne $taskValidation.scenario_sha256) { throw '계산 후 지도 설정이 변경되었습니다.' }
$taskManifest=Get-Content -LiteralPath (Join-Path $taskStaging 'run_manifest.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$taskPassengerHash=(Get-FileHash -LiteralPath (Join-Path $taskRoot 'output/gaehwasan/results.json') -Algorithm SHA256).Hash.ToLowerInvariant()
if ($taskPassengerHash -ne $taskManifest.previous_passenger_results_sha256) { throw '보존할 대피 결과가 변경되었습니다.' }
# The parser cache contains the staging directory's paths; preserve it outside
# the active run so a later raw-data read regenerates a correctly rooted cache.
$taskCache=Join-Path $taskStaging 'gaehwasan.pickle'
if (Test-Path -LiteralPath $taskCache) {
    Move-Item -LiteralPath $taskCache -Destination (Join-Path $taskRoot 'output/qa/station_fire_20261029/fdsreader_staging_cache.pickle')
}
if (Test-Path -LiteralPath $taskCurrent) { Move-Item -LiteralPath $taskCurrent -Destination $taskArchive }
try { Move-Item -LiteralPath $taskStaging -Destination $taskCurrent }
catch {
    if ((Test-Path -LiteralPath $taskArchive) -and -not (Test-Path -LiteralPath $taskCurrent)) {
        Move-Item -LiteralPath $taskArchive -Destination $taskCurrent
    }
    throw
}
$taskManifest.status='activated'
$taskManifest.run_directory='output/fds/gaehwasan'
$taskManifest | Add-Member -NotePropertyName previous_fds_archive -NotePropertyValue 'output/fds/gaehwasan_before_20261029_layout12'
$taskManifest | Add-Member -NotePropertyName activated_at_utc -NotePropertyValue ([DateTime]::UtcNow.ToString('o'))
[System.IO.File]::WriteAllText((Join-Path $taskCurrent 'run_manifest.json'),($taskManifest | ConvertTo-Json -Depth 12),[System.Text.UTF8Encoding]::new($false))
Write-Output '새 FDS 결과를 활성화했습니다. 이전 FDS 결과와 승객 대피 결과는 보존했습니다.'
