param([Parameter(Mandatory=$true)][string]$RunRoot)
$ErrorActionPreference='Stop'
$projectRoot=Split-Path $PSScriptRoot -Parent
$runPath=(Resolve-Path -LiteralPath $RunRoot).Path
$compose=Join-Path $runPath 'compose.benchmark.json'
$manifest=Get-Content -LiteralPath (Join-Path $runPath 'manifest.json') -Raw | ConvertFrom-Json
$projectName=$manifest.compose_project
$launcher=Join-Path $projectRoot 'scripts\start_local_ollama.ps1'
$variants=@(
    @{Name='initial';Backend='cuda_v13';Parallel=1;Batch=128},
    @{Name='optimized';Backend='vulkan';Parallel=2;Batch=512}
)
try{
    foreach($variant in $variants){
        $caseDir=Join-Path $runPath $variant.Name
        New-Item -ItemType Directory -Path $caseDir -Force | Out-Null
        $completed=Join-Path $caseDir 'report.json'
        if(Test-Path -LiteralPath $completed){
            $previous=Get-Content -LiteralPath $completed -Raw | ConvertFrom-Json
            if($previous.status -eq '완료'){Write-Output "$($variant.Name): 완료된 측정 보존";continue}
        }
        & $launcher -Restart -Backend $variant.Backend -Parallel $variant.Parallel -KvCacheType q8_0 -FlashAttention true
        $dockerArgs=@('compose','-f',$compose,'-p',$projectName,'run','--rm','--no-deps',
            '-e',"PERSONA_EVAC_DECISION_WORKERS=$($variant.Parallel)",
            '-e',"PERSONA_EVAC_PROMPT_BATCH_SIZE=$($variant.Batch)",
            '--entrypoint','python','benchmark','/benchmark/scripts/measure_full100.py',
            '--run-root','/measurement','--variant',$variant.Name)
        & docker @dockerArgs 2>&1 | Tee-Object -FilePath (Join-Path $caseDir ('console_'+(Get-Date -Format 'yyyyMMddTHHmmss')+'.log'))
        if($LASTEXITCODE -ne 0){throw "$($variant.Name) 100명 측정이 실패했습니다. 해당 상태 파일을 확인하세요."}
    }
    $before=Get-Content -LiteralPath (Join-Path $runPath 'initial\report.json') -Raw | ConvertFrom-Json
    $after=Get-Content -LiteralPath (Join-Path $runPath 'optimized\report.json') -Raw | ConvertFrom-Json
    if($before.code_sha256 -ne $after.code_sha256 -or $before.source_sha256 -ne $after.source_sha256 -or
       $before.config_sha256 -ne $after.config_sha256 -or $before.model_digest -ne $after.model_digest){
        throw '두 측정의 코드·입력·모델이 일치하지 않습니다.'
    }
    $report=[ordered]@{
        status='100명 비교 완료';manifest='manifest.json';initial=$before;optimized=$after
        inference_time_reduction_percent=100*(1-$after.inference_elapsed_s/$before.inference_elapsed_s)
        total_time_reduction_percent=100*(1-$after.total_processing_wall_s/$before.total_processing_wall_s)
        same_code_and_inputs_and_model=$true
        note='실제 100명 시뮬레이션 두 실행의 비교입니다. 실행기 수치 차이에 따라 이동과 판단 횟수는 달라질 수 있습니다.'
    }
    $report | ConvertTo-Json -Depth 15 | Set-Content -LiteralPath (Join-Path $runPath 'comparison.json') -Encoding utf8
    Write-Output ($report | ConvertTo-Json -Depth 4)
}finally{
    & $launcher -Restart -Backend vulkan -Parallel 2 -KvCacheType q8_0 -FlashAttention true
    & docker compose -f $compose -p $projectName down --remove-orphans
}
