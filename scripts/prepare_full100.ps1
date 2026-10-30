param([string]$RunName=(Get-Date -Format 'yyyyMMddTHHmmss'))
$ErrorActionPreference='Stop'
$projectRoot=Split-Path $PSScriptRoot -Parent
if($RunName -notmatch '^[a-zA-Z0-9_-]+$'){throw '측정 이름에는 영문·숫자·밑줄·하이픈만 사용할 수 있습니다.'}
$runPath=Join-Path $projectRoot ('output\qa\full100\'+$RunName)
if(Test-Path -LiteralPath $runPath){throw '같은 이름의 측정 폴더가 이미 있습니다.'}
$workspace=Join-Path $runPath 'workspace'
$shared=Join-Path $runPath 'shared_output'
New-Item -ItemType Directory -Path $workspace,$shared -Force | Out-Null
Copy-Item -LiteralPath (Join-Path $projectRoot 'src') -Destination (Join-Path $workspace 'src') -Recurse
Copy-Item -LiteralPath (Join-Path $projectRoot 'configs') -Destination (Join-Path $workspace 'configs') -Recurse
New-Item -ItemType Directory -Path (Join-Path $workspace 'scripts'),(Join-Path $workspace 'inputs') -Force | Out-Null
Copy-Item -LiteralPath (Join-Path $projectRoot 'scripts\measure_full100.py') -Destination (Join-Path $workspace 'scripts\measure_full100.py')
Copy-Item -LiteralPath (Join-Path $projectRoot 'output\gaehwasan\persona_source.json') -Destination (Join-Path $workspace 'inputs\persona_source.json')
$fds=Join-Path $shared 'fds\gaehwasan\derived'
New-Item -ItemType Directory -Path $fds -Force | Out-Null
foreach($name in @('fields.npz','metadata.json')){
    Copy-Item -LiteralPath (Join-Path $projectRoot ('output\fds\gaehwasan\derived\'+$name)) -Destination (Join-Path $fds $name)
}
$profileParent=Join-Path $shared 'gaehwasan'
New-Item -ItemType Directory -Path $profileParent -Force | Out-Null
Copy-Item -LiteralPath (Join-Path $projectRoot 'output\gaehwasan\private_profile_cache') -Destination (Join-Path $profileParent 'private_profile_cache') -Recurse
$image=(& docker compose images -q app | Select-Object -First 1).Trim()
if($image -notmatch '^[a-f0-9]{64}$'){throw '현재 앱의 고정 이미지 ID를 확인하지 못했습니다.'}
$composeProject='persona-full100-'+$RunName.ToLowerInvariant()
$compose=@{services=@{benchmark=@{
    image="sha256:$image";pull_policy='never';working_dir='/benchmark';entrypoint=@('python')
    environment=@{PYTHONPATH='/benchmark/src';PYTHONDONTWRITEBYTECODE='1';PYTHONUNBUFFERED='1';
        OLLAMA_BASE_URL='http://host.docker.internal:11434';OLLAMA_MODEL='gemma4:e2b-it-qat';
        LLM_PROVIDER='ollama';PERSONA_EVAC_OUTPUT='/measurement/shared_output'}
    volumes=@(
        @{type='bind';source=$workspace;target='/benchmark';read_only=$true},
        @{type='bind';source=$runPath;target='/measurement'},
        @{type='bind';source=(Join-Path $projectRoot 'output\ollama');target='/host_ollama';read_only=$true}
    )
}}}
$compose | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath (Join-Path $runPath 'compose.benchmark.json') -Encoding utf8
$hashes=[ordered]@{}
foreach($file in (Get-ChildItem -LiteralPath $workspace,$shared -Recurse -File)){
    $relative=[IO.Path]::GetRelativePath($runPath,$file.FullName).Replace('\','/')
    $hashes[$relative]=(Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
}
$protected=[ordered]@{}
foreach($name in @('output\gaehwasan\results.json','output\gaehwasan\persona_source.json',
    'data\raw\nemotron_personas_100.jsonl','data\processed\julich_features.csv',
    'configs\scenarios\gaehwasan.json','configs\fds\gaehwasan.fds',
    'output\fds\gaehwasan\derived\fields.npz','output\fds\gaehwasan\derived\metadata.json')){
    $path=Join-Path $projectRoot $name
    if(Test-Path -LiteralPath $path){$protected[$name]=(Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()}
}
$manifest=[ordered]@{status='100명 측정 준비 완료';run_name=$RunName;compose_project=$composeProject;
    image_id=$image;frozen_files_sha256=$hashes;original_files_before=$protected;
    variants=@{initial=@{backend='cuda_v13';parallel=1;prompt_batch_size=128;kv_cache_type='q8_0'};
               optimized=@{backend='vulkan';parallel=2;prompt_batch_size=512;kv_cache_type='q8_0'}};
    method='같은 코드·원본 100명·화재·600초. 정적 성향은 동일한 준비 캐시, 상황 판단은 전부 실제 추론.'}
$manifest | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath (Join-Path $runPath 'manifest.json') -Encoding utf8
Write-Output ([pscustomobject]$manifest | Select-Object status,run_name,compose_project,image_id | ConvertTo-Json)
Write-Output $runPath
