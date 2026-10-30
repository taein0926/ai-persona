"""Freeze a new-map fire run without changing passenger calculation state."""
import hashlib
import json
from datetime import datetime,timezone
from pathlib import Path

import numpy as np
from persona_evac.integration.fds_station_geometry import rasterize,write_station_fds

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'output/fds/gaehwasan_20261029_layout12'


def prepare():
    RUN.mkdir(parents=True,exist_ok=True)
    config_path=ROOT/'configs/scenarios/gaehwasan.json';raw=config_path.read_bytes()
    config=json.loads(raw);snapshot=RUN/'scenario.json'
    if snapshot.exists() and snapshot.read_bytes()!=raw:raise ValueError('기존 실행 스냅샷과 현재 지도가 다릅니다.')
    snapshot.write_bytes(raw)
    deck=ROOT/'configs/fds/gaehwasan.fds'
    audit=RUN/'geometry_audit.json'
    report=write_station_fds(config,deck,audit)
    (RUN/'gaehwasan.fds').write_bytes(deck.read_bytes())
    check=RUN/'check';check.mkdir(exist_ok=True)
    (check/'gaehwasan.fds').write_text(deck.read_text(encoding='utf8').replace('T_END=600.0','T_END=0.0'),encoding='utf8')
    grid=rasterize(config)
    np.savez_compressed(RUN/'voxel_geometry.npz',solid=grid['solid'],x0=grid['x0'],y0=grid['y0'])
    old=ROOT/'output/gaehwasan/results.json'
    manifest={'prepared_at_utc':datetime.now(timezone.utc).isoformat(),
        'scenario_sha256':hashlib.sha256(raw).hexdigest(),'deck_sha256':report['deck_sha256'],
        'previous_passenger_results_sha256':hashlib.sha256(old.read_bytes()).hexdigest(),
        'passenger_simulation_enabled':config['simulation_enabled'],'fire_duration_s':config['fire']['time_end_s'],
        'run_directory':RUN.relative_to(ROOT).as_posix(),'mpi_processes':4,'omp_threads':1,
        'status':'prepared','cells':report['cells'],'mesh_cell_m':1}
    (RUN/'run_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({**manifest,'geometry':report},ensure_ascii=False))


if __name__=='__main__':prepare()
