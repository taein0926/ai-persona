from __future__ import annotations

import os
import json
import hashlib
import threading
from uuid import uuid4
from pathlib import Path

from fastapi import FastAPI, HTTPException
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


def _results_path() -> Path:
    path = OUTPUT / "latest/results.json"
    if not path.exists():
        run_experiment(CONFIG)
    return path


def _station_source_path() -> Path:
    path = OUTPUT / "gaehwasan/persona_source.json"
    if not path.exists():
        raise HTTPException(status_code=422, detail="고유 페르소나 준비가 필요합니다: prepare-station-personas")
    source=json.loads(path.read_text(encoding='utf-8'))
    if os.environ.get('LLM_PROVIDER','ollama')=='ollama' and source.get('provenance',{}).get('provider') not in (None,'ollama'):
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
    if artifact.exists():
        result = json.loads(artifact.read_text(encoding="utf-8"))
        # A read never recomputes or replaces an active experiment after code,
        # source, policy or field changes. Preserve original provenance verbatim.
        result['artifact_status'] = {
            'implementation_matches': result.get('implementation_sha256') == _station_signature(),
            'policy_matches': policy_matches(result),
            'automatic_recomputation': False,
        }
        return result
    raise HTTPException(status_code=404, detail='저장된 대피 결과가 없습니다. 명시적으로 계산을 실행하세요.')


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
