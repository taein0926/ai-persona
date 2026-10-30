"""Export and validate this authorized fire run as soon as FDS completes."""
import json
import sys
import time
import traceback
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'output/fds/gaehwasan_20261029_layout12'


def status(phase,**extra):
    (RUN/'postprocess_status.json').write_text(json.dumps({'phase':phase,
        'updated_at_utc':datetime.now(timezone.utc).isoformat(),**extra},ensure_ascii=False,indent=2),encoding='utf8')
    print(phase,flush=True)


def finish():
    status('waiting_for_fds')
    started=time.monotonic()
    while time.monotonic()-started<7200:
        log=(RUN/'gaehwasan.out').read_text(errors='replace') if (RUN/'gaehwasan.out').exists() else ''
        if 'FDS completed successfully' in log:break
        if 'STOP: Numerical Instability' in log or 'ERROR:' in log:raise RuntimeError('FDS 실행 오류를 확인해야 합니다.')
        time.sleep(5)
    else:raise TimeoutError('FDS 완료 대기 제한 2시간을 초과했습니다.')
    from persona_evac.integration.fds_environment import export_fds
    status('exporting_fields')
    export_fds(RUN,RUN/'derived',RUN/'scenario.json',RUN/'gaehwasan.fds')
    status('validating_fields')
    sys.path.insert(0,str(ROOT/'scripts'))
    from validate_station_fire import validate
    validate()
    status('validated_ready_for_activation')


if __name__=='__main__':
    try:finish()
    except Exception as exc:
        status('failed',error=str(exc),traceback=traceback.format_exc());raise
