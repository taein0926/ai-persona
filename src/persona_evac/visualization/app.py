from __future__ import annotations

import os
import json
import hashlib
import re
import threading
from collections import OrderedDict
from copy import deepcopy
from uuid import uuid4
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.middleware.gzip import GZipMiddleware

from persona_evac.integration.pyfds_adapter import inspect_pyfds_checkout
from persona_evac.integration.runner import run_experiment
from persona_evac.schemas import ExperimentResults
from persona_evac.station_artifacts import station_signature as _station_signature,policy_matches
from persona_evac.integration.station import StationRequest as StationRequest, simulate_station as simulate_station


PACKAGE_ROOT = Path(__file__).resolve().parents[3]
ROOT = PACKAGE_ROOT if (PACKAGE_ROOT / "index.html").exists() else Path.cwd()
OUTPUT = Path(os.environ.get("PERSONA_EVAC_OUTPUT", ROOT / "output"))
CONFIG = ROOT / "configs/experiments/mvp.json"
app = FastAPI(title="Persona Evacuation MVP", version="0.1.0")
app.add_middleware(GZipMiddleware,minimum_size=1024,compresslevel=3)
STATION_LOCK = threading.Lock()
_JSON_LOCK = threading.RLock()
_JSON_FILES = OrderedDict()


def _file_version(path):
    path=Path(path)
    try:
        stat=path.stat()
    except FileNotFoundError:
        return (str(path.resolve()),None)
    return (str(path.resolve()),stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)


def _read_json(path):
    """파일별 최신 해석 결과만 최대 4개 보관한다. 내부 반환값은 수정하지 않는다."""
    path=Path(path);name=str(path.resolve())
    with _JSON_LOCK:
        for _ in range(3):
            version=_file_version(path)
            cached=_JSON_FILES.get(name)
            if cached is not None and cached[0]==version:
                _JSON_FILES.move_to_end(name)
                return cached[1]
            value=json.loads(path.read_bytes())
            if version!=_file_version(path):continue
            _JSON_FILES[name]=(version,value);_JSON_FILES.move_to_end(name)
            while len(_JSON_FILES)>4:_JSON_FILES.popitem(last=False)
            return value
    raise HTTPException(status_code=503,detail='입력 파일이 갱신 중입니다. 잠시 후 다시 조회하세요.')


@app.middleware('http')
async def refresh_local_assets(request, call_next):
    response = await call_next(request)
    path = request.url.path
    if path == '/' or path.startswith('/api/') or path.endswith(('.js','.css')):
        response.headers['Cache-Control'] = 'no-store, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
    return response


def _results_path() -> Path:
    path = OUTPUT / "latest/results.json"
    if not path.exists():
        run_experiment(CONFIG)
    return path


def _station_source_path() -> Path:
    path = OUTPUT / "gaehwasan/persona_source.json"
    if not path.exists():
        raise HTTPException(status_code=422, detail="고유 페르소나 준비가 필요합니다: prepare-station-personas")
    source=_read_json(path)
    if source.get('provenance',{}).get('provider') not in (None,'ollama'):
        raise HTTPException(status_code=422,detail='로컬 Ollama로 페르소나 성향을 다시 준비해야 합니다.')
    return path


@app.get("/api/results")
def results() -> dict[str, object]:
    path = _results_path()
    try:
        payload = ExperimentResults.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Invalid result artifact: {exc}") from exc
    data = payload.model_dump(mode="json")
    data["variants"] = {"C": data["variants"]["C"]}
    data["summaries"] = {"C": data["summaries"]["C"]}
    data["comparisons"] = []
    return data


@app.get("/api/health")
def health() -> dict[str, object]:
    return {"status": "ok", "pyfds": inspect_pyfds_checkout(ROOT / "vendor/pyFDS-Evac")}


@app.get("/")
def index(result: str | None = None):
    if result is None:
        try:
            preference=OUTPUT/'gaehwasan/display_selection.json'
            selected=_read_json(preference) if preference.exists() else {}
            run_id=selected.get('result_id','');follow=selected.get('follow_run_id','')
            query=''
            if re.fullmatch(r'[0-9]{8}T[0-9]{6}',follow):
                activation=OUTPUT/'gaehwasan/private_verification/activation.json'
                active=_read_json(activation) if activation.exists() else {}
                if active.get('run_id')==follow:run_id=follow
                else:query=f'&pending={follow}'
            if (re.fullmatch(r'[0-9]{8}T[0-9]{6}',run_id)
                    and (OUTPUT/'simulation_runs'/run_id/'results.json').is_file()):
                return RedirectResponse(f'/?result={run_id}{query}',status_code=307)
        except (OSError,ValueError,TypeError):
            pass
    return FileResponse(ROOT / "gaehwasan.html")


LOADED_SIGNATURE=_station_signature()


@app.get("/api/gaehwasan")
def gaehwasan_results() -> dict:
    artifact = OUTPUT / "gaehwasan/results.json"
    if not artifact.exists():
        raise HTTPException(status_code=404,detail={
            'code':'station_result_missing','message':'새 지도 기준 대피 계산 결과가 없습니다.'})
    result = _read_json(artifact)
    config_bytes = (ROOT/'configs/scenarios/gaehwasan.json').read_bytes()
    if (result.get('input_config_sha256') != hashlib.sha256(config_bytes).hexdigest()
            or result.get('implementation_sha256') != _station_signature()
            or not policy_matches(result)):
        raise HTTPException(status_code=409,detail={
            'code':'station_result_outdated','message':'현재 지도·계산 코드와 맞지 않는 이전 결과는 표시하지 않습니다.'})
    # Only a matching computed artifact can enter the replay path.
    return {**result,'artifact_status':{'implementation_matches':True,
        'policy_matches':True,'automatic_recomputation':False}}


@app.get('/api/gaehwasan/replay')
def gaehwasan_saved_replay() -> dict:
    """활성화 당시 검증한 저장 결과만 읽는다. 현재 계산 조건으로 바꾸지 않는다."""
    artifact = OUTPUT/'gaehwasan/results.json'
    if not artifact.exists():
        raise HTTPException(status_code=404, detail={
            'code':'station_result_missing','message':'재생할 저장 결과가 없습니다.'})
    version = _file_version(artifact)
    result = _read_json(artifact)
    activation_path = OUTPUT/'gaehwasan/private_verification/activation.json'
    try:
        activation = _read_json(activation_path)
        with artifact.open('rb') as stream:
            digest = hashlib.file_digest(stream,'sha256').hexdigest()
    except (OSError, ValueError):
        raise HTTPException(status_code=409, detail={
            'code':'station_replay_unverified','message':'저장 결과의 활성화 검증 기록을 확인할 수 없습니다.'}) from None
    if version != _file_version(artifact):
        raise HTTPException(status_code=503,detail='저장 결과가 갱신 중입니다. 잠시 후 다시 조회하세요.')
    if (not isinstance(activation,dict) or not isinstance(result,dict)
            or activation.get('artifact_sha256') != digest
            or not result.get('implementation_sha256')
            or activation.get('implementation_sha256') != result['implementation_sha256']):
        raise HTTPException(status_code=409, detail={
            'code':'station_replay_unverified','message':'저장 결과와 활성화 검증 기록이 일치하지 않습니다.'})
    scenario = result.get('scenario')
    agents = result.get('agents')
    summary = result.get('summary')
    if (not isinstance(scenario,dict) or not scenario.get('floors')
            or not isinstance(agents,list) or not agents
            or not isinstance(summary,dict) or summary.get('count') != len(agents)):
        raise HTTPException(status_code=422, detail={
            'code':'station_replay_incomplete','message':'저장 결과에 재생할 지도·인원·결과가 없습니다.'})
    config = ROOT/'configs/scenarios/gaehwasan.json'
    return {**result,'artifact_status':{
        'mode':'stored_replay','snapshot_verified':True,'artifact_sha256':digest,
        'implementation_matches':result['implementation_sha256'] == _station_signature(),
        'config_matches':result.get('input_config_sha256') == hashlib.sha256(config.read_bytes()).hexdigest(),
        'policy_matches':policy_matches(result),'automatic_recomputation':False}}


@app.get('/api/gaehwasan/history/{run_id}')
def gaehwasan_history(run_id: str) -> dict:
    """지정한 저장 실행만 재생한다. 현재 활성 파일과 실행 중인 계산은 바꾸지 않는다."""
    if not re.fullmatch(r'[0-9]{8}T[0-9]{6}',run_id):
        raise HTTPException(status_code=404,detail='저장 실행을 찾을 수 없습니다.')
    directory=OUTPUT/'simulation_runs'/run_id;artifact=directory/'results.json';report_path=directory/'report.json'
    if not artifact.is_file() or not report_path.is_file():
        raise HTTPException(status_code=404,detail='완료된 저장 결과가 없습니다.')
    version=_file_version(artifact)
    result=_read_json(artifact);report=_read_json(report_path)
    with artifact.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    if version!=_file_version(artifact):
        raise HTTPException(status_code=503,detail='저장 결과가 갱신 중입니다.')
    if digest!=report.get('result_sha256') or result.get('summary')!=report.get('summary'):
        raise HTTPException(status_code=409,detail='저장 결과와 완료 기록이 일치하지 않습니다.')
    config=ROOT/'configs/scenarios/gaehwasan.json'
    return {**result,'artifact_status':{
        'mode':'stored_replay','snapshot_verified':True,'artifact_sha256':digest,'history_run_id':run_id,
        'implementation_matches':result.get('implementation_sha256')==_station_signature(),
        'config_matches':config.exists() and result.get('input_config_sha256')==hashlib.sha256(config.read_bytes()).hexdigest(),
        'policy_matches':policy_matches(result),'automatic_recomputation':False}}


@app.get('/api/gaehwasan/activation')
def gaehwasan_activation() -> dict:
    """현재 화면에 연결한 별도 계산의 반영 정보만 읽는다. 계산을 시작하지 않는다."""
    pending=OUTPUT/'gaehwasan/pending_run.json'
    activation=OUTPUT/'gaehwasan/private_verification/activation.json'
    try:run=_read_json(pending) if pending.exists() else None
    except (OSError,ValueError):run=None
    try:active=_read_json(activation) if activation.exists() else {}
    except (OSError,ValueError):active={}
    return {'run':run,'active_artifact_sha256':active.get('artifact_sha256')}


@app.get('/api/gaehwasan/setup')
def gaehwasan_setup(agent_count: Annotated[int,Query(ge=10,le=100)] = 100) -> dict:
    """Return real simulation inputs, without constructing a result or advancing time."""
    from persona_evac.integration.station import initialize_station
    from persona_evac.integration.station_layout import attach_display_geometry
    from persona_evac.integration.fds_environment import geometry_hash,sha

    config = _read_json(ROOT/'configs/scenarios/gaehwasan.json')
    source_path = _station_source_path()
    source = _read_json(source_path)
    setup = initialize_station(source,config,StationRequest(agent_count=agent_count),preview_only=True)
    _attach_prepared_profiles(setup,source,source_path)
    attach_display_geometry(setup['scenario'])
    setup['calculation_available'] = config.get('simulation_enabled') is not False
    meta_path = OUTPUT/'fds/gaehwasan/derived/metadata.json'
    if meta_path.exists():
        meta = _read_json(meta_path)
        if (meta.get('geometry_sha256') == geometry_hash(config)
                and meta.get('deck_sha256') == sha(ROOT/'configs/fds/gaehwasan.fds')
                and (meta_path.parent/'fields.npz').exists()):
            setup['fire_precomputation'] = {'status':'ready','duration_s':meta['time_end_s'],
                'mesh_cell_m':meta['mesh_cell_m'],
                'gas_exposure_ready':all(key in meta.get('quantities',{}) for key in ('co','co2','o2'))}
    return setup


def _attach_prepared_profiles(setup,source,source_path):
    """원본·정책·모델이 일치하는 준비 성향만 읽는다. 조회 중 추론하지 않는다."""
    from persona_evac.agent_policy import POLICY
    from persona_evac.llm.ollama import DEFAULT_MODEL
    from persona_evac.persona.full_profile import CONFIG,ProfileFormatError,validate_full_profile

    path=OUTPUT/'gaehwasan/profile_preparations/full_persona'/CONFIG['version']/'profiles.json'
    try:
        prepared=_read_json(path)
        if (prepared.get('classification_method')!='full_persona'
                or prepared.get('version')!=CONFIG['version']
                or prepared.get('source_sha256')!=hashlib.sha256(source_path.read_bytes()).hexdigest()):
            return
        provenance=source['provenance']
        model={'name':provenance.get('model'),
               'digest':provenance.get('mapping_settings',{}).get('model_digest')}
        if model['name']!=os.environ.get('OLLAMA_MODEL',DEFAULT_MODEL) or not model['digest']:
            return
        rows=prepared['profiles'];profiles={row['agent_id']:row['profile'] for row in rows}
        if len(profiles)!=len(rows):return
        selected=[]
        for agent in setup['agents']:
            if agent.get('classification_pending'):return
            profile=profiles[agent['source_agent_id']]
            validate_full_profile(profile,agent['persona'],POLICY['experiment_seed'])
            if profile['model']!=model:return
            selected.append((agent,profile))
    except (OSError,ValueError,KeyError,TypeError,ProfileFormatError):
        return
    for agent,profile in selected:
        agent['experimental_profile']=deepcopy(profile)


@app.post("/api/gaehwasan")
def gaehwasan_run(request: StationRequest) -> dict:
    if _station_signature()!=LOADED_SIGNATURE:
        raise HTTPException(status_code=409,detail='계산 코드가 변경되었습니다. 앱을 재시작한 뒤 계산하세요.')
    if not STATION_LOCK.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="다른 대피 계산이 진행 중입니다.")
    try:
        config_path = ROOT / "configs/scenarios/gaehwasan.json"
        source_text = _station_source_path().read_bytes()
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get('simulation_enabled') is False:
            raise HTTPException(status_code=409,detail='시뮬레이션 계산은 보류 중입니다. 명시적으로 계산을 재개한 뒤 실행할 수 있습니다.')
        try:
            result = simulate_station(json.loads(source_text),config,request)
        except (ValueError,RuntimeError) as exc:
            raise HTTPException(status_code=422,detail=str(exc)) from exc
        result["source"]["artifact_sha256"] = hashlib.sha256(source_text).hexdigest()
        result["input_config_sha256"] = hashlib.sha256(config_path.read_bytes()).hexdigest()
        result["implementation_sha256"] = _station_signature()
        if result['implementation_sha256']!=LOADED_SIGNATURE:
            raise HTTPException(status_code=409,detail='계산 도중 코드가 변경되어 결과를 저장하지 않았습니다.')
        result['computation']={'code_sha256':LOADED_SIGNATURE,'kind':'recomputed_by_api'}
        path = OUTPUT / "gaehwasan/results.json"
        path.parent.mkdir(parents=True,exist_ok=True)
        temp=path.with_suffix(f".{uuid4().hex}.tmp")
        temp.write_text(json.dumps(result,ensure_ascii=False),encoding="utf-8")
        temp.replace(path)
        return result
    finally:
        STATION_LOCK.release()


@app.get("/t-junction")
def original_view() -> FileResponse:
    return FileResponse(ROOT / "index.html")


@app.get("/{asset_name}")
def asset(asset_name: str) -> FileResponse:
    if asset_name not in {"app.js", "styles.css", "overrides.css", "station_viewer.js", "private_inspector.js", "station_base.css", "gaehwasan.js", "gaehwasan.css", "gaehwasan3d.js"}:
        raise HTTPException(status_code=404)
    return FileResponse(ROOT / asset_name)


@app.get('/vendor/three/{asset}')
def three_asset(asset:str):
    if asset not in {'three.module.js','three.core.js','OrbitControls.js','LICENSE'}:
        raise HTTPException(status_code=404)
    return FileResponse(ROOT/'vendor/three'/asset)
