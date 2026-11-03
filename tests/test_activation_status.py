import importlib.util,json
from pathlib import Path
from fastapi.testclient import TestClient

def test_activation_status_reads_metadata_only_and_tracks_replacement(tmp_path):
    staged=tmp_path/'source/persona_evac/visualization/app.py';staged.parent.mkdir(parents=True)
    staged.write_bytes(Path('src/persona_evac/visualization/app.py').read_bytes())
    spec=importlib.util.spec_from_file_location('activation_test_app',staged)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.OUTPUT=tmp_path
    def forbidden(*args,**kwargs):raise AssertionError('반영 정보 조회에서 계산을 호출했습니다.')
    module.simulate_station=forbidden;module._station_source_path=forbidden
    with TestClient(module.app) as client:
        assert client.get('/api/gaehwasan/activation').json()=={'run':None,'active_artifact_sha256':None}
        pending=tmp_path/'gaehwasan/pending_run.json';pending.parent.mkdir()
        pending.write_text(json.dumps({'run_id':'20261103T100000','status':'계산 중'}))
        active=tmp_path/'gaehwasan/private_verification/activation.json';active.parent.mkdir()
        active.write_text(json.dumps({'artifact_sha256':'old'}))
        assert client.get('/api/gaehwasan/activation').json()['active_artifact_sha256']=='old'
        temp=pending.with_suffix('.tmp');temp.write_text(json.dumps({'run_id':'20261103T100000','status':'화면 반영·검증 중','artifact_sha256':'new'}));temp.replace(pending)
        active.write_text(json.dumps({'artifact_sha256':'new'}))
        response=client.get('/api/gaehwasan/activation').json()
        assert response['run']['artifact_sha256']==response['active_artifact_sha256']=='new'
