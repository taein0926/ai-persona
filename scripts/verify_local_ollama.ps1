param([string]$Model='gemma4:e2b-it-qat',[switch]$Warmup,[ValidateSet(1,2)][int]$ExpectedParallel=2,[string]$ReportPath='')
$ErrorActionPreference='Stop'
$taskProject=Split-Path $PSScriptRoot -Parent
$taskRuntime=Get-Content -LiteralPath (Join-Path $taskProject 'output/ollama/runtime_config.json') -Raw | ConvertFrom-Json
$taskOwner=@(Get-NetTCPConnection -LocalPort 11434 -State Listen | Select-Object -ExpandProperty OwningProcess -Unique)
if($taskOwner.Count -ne 1 -or $taskOwner[0] -ne $taskRuntime.server_pid){throw '실제 서버 PID와 프로젝트 실행 기록이 다릅니다. 시작 스크립트로 다시 실행하세요.'}
if($Warmup){
    $taskBody=@{model=$Model;stream=$false;think=$false;keep_alive='30m';format=@{type='object';properties=@{ready=@{type='boolean'}};required=@('ready');additionalProperties=$false};options=@{num_ctx=4096;num_predict=64;num_batch=512;temperature=0;seed=20260916};messages=@(@{role='user';content='준비 확인입니다. ready=true를 JSON으로 반환하세요.'})} | ConvertTo-Json -Depth 8
    $null=Invoke-RestMethod 'http://127.0.0.1:11434/api/chat' -Method Post -ContentType 'application/json; charset=utf-8' -Body ([Text.Encoding]::UTF8.GetBytes($taskBody)) -TimeoutSec 180
}
$taskModels=@((Invoke-RestMethod 'http://127.0.0.1:11434/api/ps').models | Where-Object {$_.name -eq $Model})
if($taskModels.Count -ne 1 -or $taskModels[0].size_vram -lt 500MB){throw '요청한 모델의 GPU 적재를 확인하지 못했습니다.'}
$taskShow=Invoke-RestMethod 'http://127.0.0.1:11434/api/show' -Method Post -ContentType 'application/json' -Body (@{model=$Model} | ConvertTo-Json)
$taskBlob=[regex]::Match($taskShow.modelfile,'(?m)^FROM\s+"?([^"\r\n]+)').Groups[1].Value.Trim()
$taskRunners=@(Get-CimInstance Win32_Process -Filter "ParentProcessId=$($taskRuntime.server_pid)" | Where-Object {$_.Name -eq 'llama-server.exe' -and $_.CommandLine.Contains($taskBlob)})
if($taskRunners.Count -ne 1){throw '모델에 해당하는 실제 추론 프로세스를 찾지 못했습니다.'}
$taskCommand=$taskRunners[0].CommandLine
if($taskCommand -notmatch "-np $ExpectedParallel\b" -or $taskCommand -notmatch '--cache-type-k q8_0\b' -or $taskCommand -notmatch '--cache-type-v q8_0\b' -or $taskCommand -notmatch '--flash-attn on\b'){throw '실제 추론 옵션이 요청한 슬롯 수·q8_0·Flash Attention 고정 설정과 다릅니다.'}
$taskExpectedContext=4096*$ExpectedParallel
if($taskCommand -notmatch "-c $taskExpectedContext\b" -or $taskModels[0].context_length -ne 4096){throw '실제 문맥 길이가 요청당 4096토큰과 다릅니다.'}
$taskLog=Get-Content -LiteralPath (Join-Path $taskProject 'output/ollama/local_server.stderr.log') -Raw
$taskLogStart=$taskLog.LastIndexOf('msg="starting llama-server"')
if($taskLogStart -lt 0){throw '실제 추론 프로세스의 시작 로그를 확인하지 못했습니다.'}
$taskLast=$taskLog.Substring($taskLogStart)
if($taskLast -notmatch 'using device Vulkan\d+ \(NVIDIA' -or $taskLast -notmatch 'offloaded (\d+)/\1 layers to GPU' -or $taskLast -notmatch "n_seq_max\s*=\s*$ExpectedParallel\b"){
    throw '실제 NVIDIA Vulkan 전체 층 적재와 요청한 슬롯 수를 확인하지 못했습니다.'
}
$taskProof=@{status='실제 실행 설정 검증 통과';verified_at=(Get-Date).ToString('o');model=$Model;model_digest=$taskModels[0].digest;server_pid=$taskRuntime.server_pid;runner_pid=$taskRunners[0].ProcessId;command_line=$taskCommand;profile=$taskRuntime.profile;actual_parallel=$ExpectedParallel;loaded_model=$taskModels[0];evidence=@($taskLast -split "`n" | Where-Object {$_ -match 'using device|offloaded|n_seq_max|flash_attn|KV buffer'})}
if(-not $ReportPath){$ReportPath=Join-Path $taskProject 'output/ollama/runtime_verified.json'}
$taskProof | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $ReportPath -Encoding utf8
Write-Output "실제 실행 검증 통과: $Model, NVIDIA Vulkan, 전체 층 GPU 적재, 동시 요청 $ExpectedParallel 개, q8_0, Flash Attention"
