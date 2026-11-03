'use strict';
async function loadPublishedReplay(){
  const options={credentials:'omit',redirect:'error'};
  const response=await fetch('./data/manifest.json',options);
  if(!response.ok)throw new Error('저장 결과 목록을 불러오지 못했습니다.');
  const manifest=await response.json();
  if(manifest.version!==1||manifest.file!=='replay.json.gz'||manifest.payload_bytes>128*1024*1024)
    throw new Error('지원하지 않는 저장 결과 형식입니다.');
  const result=await fetch('./data/'+manifest.file,options);
  if(!result.ok)throw new Error('저장 결과를 내려받지 못했습니다.');
  const bytes=new Uint8Array(await result.arrayBuffer());
  const digest=async data=>Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',data)),n=>n.toString(16).padStart(2,'0')).join('');
  let payload=bytes;
  if(bytes[0]===31&&bytes[1]===139){
    if(bytes.length!==manifest.compressed_bytes||await digest(bytes)!==manifest.compressed_sha256)
      throw new Error('다운로드한 저장 결과의 무결성을 확인하지 못했습니다. 새로고침해 주세요.');
    if(typeof DecompressionStream==='undefined')throw new Error('최신 Chrome·Edge·Firefox·Safari에서 열어 주세요.');
    document.getElementById('message').textContent='저장 결과 확인 중';
    payload=new Uint8Array(await new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer());
  }
  if(payload.length!==manifest.payload_bytes||await digest(payload)!==manifest.payload_sha256)
    throw new Error('저장 결과의 내용이 배포 기록과 일치하지 않습니다.');
  const data=JSON.parse(new TextDecoder().decode(payload));
  if(data.agents?.length!==manifest.summary.count||!data.summary||!data.fire)
    throw new Error('저장 결과의 지도·인원 정보가 올바르지 않습니다.');
  data.calculation_available=false;
  data.artifact_status={mode:'static_replay',snapshot_verified:true,artifact_sha256:manifest.source_artifact_sha256};
  return data;
}
