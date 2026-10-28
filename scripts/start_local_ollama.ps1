param([switch]$Restart,[ValidateRange(1,16)][int]$Parallel=2,[ValidateSet('q8_0','f16')][string]$KvCacheType='q8_0',[ValidateSet('true','false')][string]$FlashAttention='true',[ValidateSet('auto','cuda_v12','cuda_v13','vulkan')][string]$Backend='vulkan')
$ErrorActionPreference='Stop'
$projectRoot=Split-Path $PSScriptRoot -Parent
$ollamaExe=(Get-Command ollama -ErrorAction Stop).Source
$ollamaDir=Split-Path $ollamaExe -Parent
$running=@(Get-Process -Name 'ollama','ollama app' -ErrorAction SilentlyContinue | Where-Object {$_.Path -and $_.Path.StartsWith($ollamaDir,[StringComparison]::OrdinalIgnoreCase)})
if($running.Count -and -not $Restart){Write-Output 'Ollama가 이미 실행 중입니다. 병렬 처리·캐시 설정을 적용해 재시작하려면 -Restart를 사용하세요.';exit 0}
if($Restart -and $running.Count){$running | Stop-Process -Force;$running | Wait-Process -Timeout 10 -ErrorAction SilentlyContinue}
$logDir=Join-Path $projectRoot 'output\ollama'
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$previousLogs=@('local_server.stdout.log','local_server.stderr.log' | ForEach-Object {Join-Path $logDir $_} | Where-Object {Test-Path -LiteralPath $_})
if($previousLogs.Count){
    $historyDir=Join-Path $logDir ('history\'+(Get-Date -Format 'yyyyMMddTHHmmssfff'))
    New-Item -ItemType Directory -Path $historyDir -Force | Out-Null
    $previousLogs | ForEach-Object {Copy-Item -LiteralPath $_ -Destination $historyDir}
}
$serverEnv=@{LLAMA_ARG_CACHE_RAM='0';OLLAMA_HOST='127.0.0.1:11434';OLLAMA_NUM_PARALLEL="$Parallel";OLLAMA_KV_CACHE_TYPE=$KvCacheType;OLLAMA_FLASH_ATTENTION=$FlashAttention;OLLAMA_LLM_LIBRARY=''}
if($Backend -ne 'auto'){$serverEnv['OLLAMA_LLM_LIBRARY']=$Backend}
$server=Start-Process -FilePath $ollamaExe -ArgumentList 'serve' -WindowStyle Hidden -Environment $serverEnv -RedirectStandardOutput (Join-Path $logDir 'local_server.stdout.log') -RedirectStandardError (Join-Path $logDir 'local_server.stderr.log') -PassThru
$ready=$false
for($attempt=0;$attempt -lt 20;$attempt++){
    if($server.HasExited){throw 'Ollama가 시작 직후 종료되었습니다. output/ollama의 로그를 확인하세요.'}
    try{$null=Invoke-RestMethod 'http://127.0.0.1:11434/api/tags' -TimeoutSec 1;$ready=$true;break}catch{Start-Sleep -Milliseconds 250}
}
if(-not $ready){throw 'Ollama의 시작 응답을 확인하지 못했습니다. output/ollama의 로그를 확인하세요.'}
$runtime=@{version=1;base_url='http://127.0.0.1:11434';server_pid=$server.Id;started_at=(Get-Date).ToString('o');profile=@{backend=$Backend;parallel=$Parallel;kv_cache_type=$KvCacheType;flash_attention=($FlashAttention -eq 'true')}}
$runtime | ConvertTo-Json -Depth 3 | Set-Content -LiteralPath (Join-Path $logDir 'runtime_config.json') -Encoding utf8
Write-Output "로컬 Ollama 시작: PID $($server.Id), 실행기 $Backend, 동시 요청 $Parallel 개, 내부 RAM 프롬프트 캐시 0, KV 캐시 $KvCacheType, Flash Attention $FlashAttention, 응답 파일 캐시 유지"
