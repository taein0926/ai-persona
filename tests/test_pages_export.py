"""정적 배포가 계산 표본을 바꾸거나 내부 자료를 공개하지 않는지 검사한다."""
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import shutil

import pytest

from scripts.build_pages import ASSETS, PUBLISHED_FILES, build, project_replay

REPO=Path(__file__).resolve().parents[1]


def recorded():
    return {'implementation_sha256':'recorded-code','scenario':{'floors':[]},
        'summary':{'count':1,'completed_count':0,'total_time_s':3.125},
        'fire':{'enabled':True,'frames':[{'time_s':.125,'k':[.012345678901]}]},
        'agents':[{'agent_id':1,'segments':[{'start':.1,'end':.25,'from':{'x':.1234567890123},
                       'native_motion':[[.125,3.1234567890123]]}],
                   'known_map':{'unused':'internal map'},'provenance':{'private':'local metadata'},
                   'people_observations':[{'time_s':1.25,'floor':'B1','people':[{'id':3},{'id':8}]}],
                   'observations':[{'id':'O1','kind':'local_hazard','time_s':.1,'perceived':{'smoke':'light'}},
                                   {'id':'O2','kind':'space_seen','time_s':.2,'new_cells':[[1,2]]}]}],
        'decisions':[{'agent_id':1,'time_s':.125,'priority':'risk','cache_key':'internal hash',
                      'observation':{'private':'local metadata'}}]}


def fixture(tmp_path):
    root=tmp_path/'project';root.mkdir()
    for name in (*ASSETS,'gaehwasan.html'):
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(REPO/name,path)
    source=root/'recorded.json';raw=json.dumps(recorded()).encode();source.write_bytes(raw)
    activation=root/'activation.json'
    activation.write_text(json.dumps({'activated':True,'artifact_sha256':hashlib.sha256(raw).hexdigest(),
                                      'implementation_sha256':'recorded-code'}))
    return root,source,activation


def test_projection_preserves_values_and_consumed_observations_without_internal_records():
    original=recorded();before=deepcopy(original);data=project_replay(original)
    assert data['summary']==original['summary'] and data['fire']==original['fire']
    assert data['agents'][0]['segments']==original['agents'][0]['segments']
    assert data['decisions'][0]['time_s']==original['decisions'][0]['time_s']
    assert data['agents'][0]['people_observations'][0]['people_count']==2
    assert data['agents'][0]['observations'][0]==original['agents'][0]['observations'][0]
    assert 'known_map' not in data['agents'][0] and 'provenance' not in data['agents'][0]
    assert 'cache_key' not in data['decisions'][0] and 'observation' not in data['decisions'][0]
    assert original==before


def test_export_is_read_only_and_subdirectory_safe_with_verified_compressed_data(tmp_path):
    root,source,activation=fixture(tmp_path);out=tmp_path/'site'
    info=build(root,out,source,activation)
    compressed=(out/'data/replay.json.gz').read_bytes();payload=gzip.decompress(compressed)
    assert hashlib.sha256(compressed).hexdigest()==info['compressed_sha256']
    assert hashlib.sha256(payload).hexdigest()==info['payload_sha256']
    assert json.loads(payload)['summary']==recorded()['summary']
    assert {p.relative_to(out).as_posix() for p in out.rglob('*') if p.is_file()}==PUBLISHED_FILES
    html=(out/'index.html').read_text(encoding='utf8')
    assert 'data-static-replay="true"' in html and 'Content-Security-Policy' in html
    assert 'src="/' not in html and 'href="/' not in html
    assert "from './vendor/three/" in (out/'gaehwasan3d.js').read_text(encoding='utf8')
    assert not (out/'t-junction').exists() and 'T자 복도' not in html
    assert '계산 결과 재생<br>' not in html and '설치 없이 재생' not in html


def test_unverified_result_and_unexpected_public_file_are_rejected(tmp_path):
    root,source,activation=fixture(tmp_path);out=tmp_path/'site'
    activation.write_text('{}')
    with pytest.raises(ValueError,match='활성화'):build(root,out,source,activation)
    assert not out.exists()
    out.mkdir();(out/'.env').write_text('must not publish')
    with pytest.raises(ValueError,match='허용하지 않은 파일'):build(root,out,source,activation)
