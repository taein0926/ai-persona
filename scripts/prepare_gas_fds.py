"""기존 FDS·승객 결과를 보존하고 가스 농도 출력이 있는 새 실행을 준비한다."""
import json
from pathlib import Path

from persona_evac.integration.fds_station_geometry import write_station_fds

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/'output/fds/gaehwasan_20261102_gas'


def prepare():
    RUN.mkdir(parents=True,exist_ok=True)
    snapshot=RUN/'scenario.json'
    raw=(ROOT/'configs/scenarios/gaehwasan.json').read_bytes()
    if snapshot.exists() and snapshot.read_bytes()!=raw:
        raise ValueError('가스 계산의 기존 지도 스냅샷과 현재 지도가 다릅니다.')
    snapshot.write_bytes(raw)
    report=write_station_fds(json.loads(raw),RUN/'gaehwasan.fds',RUN/'geometry_audit.json')
    check=RUN/'check';check.mkdir(exist_ok=True)
    (check/'gaehwasan.fds').write_text((RUN/'gaehwasan.fds').read_text(encoding='utf8').replace('T_END=600.0','T_END=0.0'),encoding='utf8')
    print(json.dumps({'run_directory':str(RUN),'deck_sha256':report['deck_sha256'],'cells':report['cells']},ensure_ascii=False))


if __name__=='__main__':prepare()
