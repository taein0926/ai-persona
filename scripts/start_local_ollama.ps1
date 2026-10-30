param([switch]$Restart,[ValidateRange(1,16)][int]$Parallel=2,[ValidateSet('q8_0','f16')][string]$KvCacheType='q8_0',[ValidateSet('true','false')][string]$FlashAttention='true',[ValidateSet('auto','cuda_v12','cuda_v13','vulkan')][string]$Backend='vulkan',[ValidatePattern('^$|^\d+(,\d+)*$')][string]$VulkanDevice='')
$ErrorActionPreference='Stop'
$projectRoot=Split-Path $PSScriptRoot -Parent
$ollamaExe=(Get-Command ollama -ErrorAction Stop).Source
$ollamaDir=Split-Path $ollamaExe -Parent
$running=@(Get-Process -Name 'ollama','ollama app' -ErrorAction SilentlyContinue | Where-Object {$_.Path -and $_.Path.StartsWith($ollamaDir,[StringComparison]::OrdinalIgnoreCase)})
if($running.Count -and -not $Restart){
    $taskExistingPath=Join-Path $projectRoot 'output/ollama/runtime_config.json'
    if(-not (Test-Path -LiteralPath $taskExistingPath)){throw '실행 중인 Ollama의 프로젝트 설정 기록이 없습니다. -Restart로 고정 설정을 적용하세요.'}
    $taskExisting=Get-Content -LiteralPath $taskExistingPath -Raw | ConvertFrom-Json
    $taskOwners=@(Get-NetTCPConnection -LocalPort 11434 -State Listen -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique)
    if($taskOwners.Count -ne 1 -or $taskOwners[0] -ne $taskExisting.server_pid -or
       $taskExisting.profile.backend -ne $Backend -or $taskExisting.profile.parallel -ne $Parallel -or
       $taskExisting.profile.kv_cache_type -ne $KvCacheType -or $taskExisting.profile.flash_attention -ne ($FlashAttention -eq 'true') -or
       ($PSBoundParameters.ContainsKey('VulkanDevice') -and $taskExisting.vulkan_visible_devices -ne $VulkanDevice)){
        throw '실제 서버 PID 또는 요청한 실행 설정이 프로젝트 기록과 다릅니다. -Restart로 다시 실행하세요.'
    }
    Write-Output '기존 프로젝트 서버 PID·실행 설정 확인 완료. 긴 추론 전 verify_local_ollama.ps1로 모델의 실제 GPU 적재·슬롯을 검사하세요.'
    exit 0
}
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
# Vulkan 번호는 재부팅·프로세스마다 달라질 수 있으므로 이전 번호를 재사용하지 않는다.
# 기본값은 매번 NVIDIA 장치를 탐색하며, 아래 실제 장치 확인을 통과해야 한다.
if($Backend -eq 'vulkan' -and -not $VulkanDevice){
    $taskProbeDir=Join-Path $ollamaDir 'lib/ollama/vulkan'
    $taskProbeExe=Join-Path $ollamaDir 'lib/ollama/llama-server.exe'
    Push-Location $taskProbeDir
    try{$taskDevices=(& $taskProbeExe --list-devices 2>&1 | Out-String)}finally{Pop-Location}
    $taskNvidia=[regex]::Matches($taskDevices,'Vulkan(\d+): NVIDIA[^\r\n]+')
    if($taskNvidia.Count -ne 1){throw 'Vulkan 장치 목록에서 NVIDIA GPU 한 개를 확인하지 못했습니다. 장치를 확인한 뒤 -VulkanDevice로 지정하세요.'}
    $VulkanDevice=$taskNvidia[0].Groups[1].Value
}
if($Backend -ne 'auto'){$serverEnv['OLLAMA_LLM_LIBRARY']=$Backend}
if($Backend -eq 'vulkan'){$serverEnv['OLLAMA_VULKAN']='true';$serverEnv['GGML_VK_VISIBLE_DEVICES']=$VulkanDevice;$serverEnv['OLLAMA_IGPU_ENABLE']='0'}
$server=Start-Process -FilePath $ollamaExe -ArgumentList 'serve' -WindowStyle Hidden -Environment $serverEnv -RedirectStandardOutput (Join-Path $logDir 'local_server.stdout.log') -RedirectStandardError (Join-Path $logDir 'local_server.stderr.log') -PassThru
$ready=$false
for($attempt=0;$attempt -lt 40;$attempt++){
    if($server.HasExited){throw 'Ollama가 시작 직후 종료되었습니다. output/ollama의 로그를 확인하세요.'}
    try{
        $null=Invoke-RestMethod 'http://127.0.0.1:11434/api/tags' -TimeoutSec 1
        $taskDiscovery=@(Get-Content -LiteralPath (Join-Path $logDir 'local_server.stderr.log') | Where-Object {$_ -match 'inference compute' -and $_ -match 'library=Vulkan' -and $_ -match 'description="NVIDIA'})
        if($Backend -ne 'vulkan' -or $taskDiscovery.Count){$ready=$true;break}
    }catch{}
    Start-Sleep -Milliseconds 250
}
if(-not $ready){Stop-Process -Id $server.Id -Force -ErrorAction SilentlyContinue;throw 'Ollama 응답 또는 실제 NVIDIA Vulkan 장치를 확인하지 못해 중단했습니다. output/ollama의 로그를 확인하세요.'}
$runtime=@{version=1;base_url='http://127.0.0.1:11434';server_pid=$server.Id;started_at=(Get-Date).ToString('o');profile=@{backend=$Backend;parallel=$Parallel;kv_cache_type=$KvCacheType;flash_attention=($FlashAttention -eq 'true')}}
if($Backend -eq 'vulkan'){$runtime['vulkan_visible_devices']=$VulkanDevice;$runtime['device_discovery_verified']=$taskDiscovery}
$runtime | ConvertTo-Json -Depth 3 | Set-Content -LiteralPath (Join-Path $logDir 'runtime_config.json') -Encoding utf8
Write-Output "로컬 Ollama 시작: PID $($server.Id), 실행기 $Backend, 동시 요청 $Parallel 개, 내부 RAM 프롬프트 캐시 0, KV 캐시 $KvCacheType, Flash Attention $FlashAttention, 응답 파일 캐시 유지"
