import importlib.util,json,hashlib
from pathlib import Path
from fastapi.testclient import TestClient

def test_history_uses_requested_artifact_without_changing_active_result(tmp_path):
    app_path=tmp_path/'source/persona_evac/visualization/app.py';app_path.parent.mkdir(parents=True)
    app_path.write_bytes(Path('src/persona_evac/visualization/app.py').read_bytes())
    spec=importlib.util.spec_from_file_location('history_app',app_path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.OUTPUT=tmp_path/'output';module.ROOT=tmp_path;module._station_signature=lambda:'new-code';module.policy_matches=lambda r:False
    module.simulate_station=lambda *a,**kw:(_ for _ in ()).throw(AssertionError('조회 중 계산 호출'))
    directory=module.OUTPUT/'simulation_runs/20261102T153027';directory.mkdir(parents=True)
    result={'summary':{'count':100,'completed_count':93,'total_time_s':400},'implementation_sha256':'old-code'}
    file=directory/'results.json';file.write_text(json.dumps(result));digest=hashlib.sha256(file.read_bytes()).hexdigest()
    report=directory/'report.json';report.write_text(json.dumps({'summary':result['summary'],'result_sha256':digest}))
    active=module.OUTPUT/'gaehwasan/results.json';active.parent.mkdir();active.write_text('current-active')
    before=file.read_bytes(),active.read_bytes()
    with TestClient(module.app) as client:
        response=client.get('/api/gaehwasan/history/20261102T153027')
        assert response.status_code==200 and response.json()['summary']['completed_count']==93
        assert response.json()['artifact_status']['history_run_id']=='20261102T153027'
        assert response.json()['artifact_status']['automatic_recomputation'] is False
        assert client.get('/api/gaehwasan/history/not-a-run').status_code==404
        report.write_text(json.dumps({'summary':result['summary'],'result_sha256':'wrong'}))
        assert client.get('/api/gaehwasan/history/20261102T153027').status_code==409
    assert (file.read_bytes(),active.read_bytes())==before
