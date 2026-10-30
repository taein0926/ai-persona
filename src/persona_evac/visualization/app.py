from __future__ import annotations

import os
import json
import hashlib
import threading
from collections import OrderedDict
from copy import deepcopy
from uuid import uuid4
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
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
def index() -> FileResponse:
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
                'mesh_cell_m':meta['mesh_cell_m']}
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
