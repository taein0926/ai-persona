import importlib.util,json
from pathlib import Path
from fastapi.testclient import TestClient

def test_default_view_tracks_selected_history_then_completed_pending_run(tmp_path):
    source=Path('src/persona_evac/visualization/app.py')
    target=tmp_path/'source/persona_evac/visualization/app.py';target.parent.mkdir(parents=True);target.write_bytes(source.read_bytes())
    spec=importlib.util.spec_from_file_location('default_app',target);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.OUTPUT=tmp_path/'output';module.ROOT=tmp_path;(tmp_path/'gaehwasan.html').write_text('saved view')
    out=module.OUTPUT/'gaehwasan';out.mkdir(parents=True)
    old=module.OUTPUT/'simulation_runs/20261102T153027/results.json';old.parent.mkdir(parents=True);old.write_text('{}')
    (out/'display_selection.json').write_text(json.dumps({'result_id':'20261102T153027','follow_run_id':'20261103T094404'}))
    active=out/'private_verification/activation.json';active.parent.mkdir();active.write_text(json.dumps({'run_id':'20261102T181500'}))
    with TestClient(module.app) as client:
        response=client.get('/',follow_redirects=False)
        assert response.status_code==307 and response.headers['location']=='/?result=20261102T153027&pending=20261103T094404'
        assert client.get('/?result=20261102T181500').status_code==200
        new=module.OUTPUT/'simulation_runs/20261103T094404/results.json';new.parent.mkdir(parents=True);new.write_text('{}')
        active.write_text(json.dumps({'run_id':'20261103T094404'}))
        response=client.get('/',follow_redirects=False)
        assert response.headers['location']=='/?result=20261103T094404'
        assert old.read_text()=='{}'
