"""Inspect the running FDS clock and estimate remaining wall time."""
import csv
import json
from datetime import datetime,timezone
from pathlib import Path

RUN=Path('output/fds/gaehwasan_20261029_layout12')


def progress():
    manifest=json.loads((RUN/'run_manifest.json').read_text(encoding='utf8'))
    rows=[]
    path=RUN/'gaehwasan_steps.csv'
    if path.exists():
        for row in list(csv.reader(path.read_text(encoding='utf8').splitlines()))[2:]:
            try:rows.append((datetime.fromisoformat(row[1].strip()),float(row[3])))
            except (ValueError,IndexError):pass
    now=datetime.now(timezone.utc);end=manifest['fire_duration_s']
    current=rows[-1][1] if rows else 0
    anchor=next((r for r in rows if (rows[-1][0]-r[0]).total_seconds()<=120),rows[0]) if rows else None
    rate=(current-anchor[1])/max(1,(rows[-1][0]-anchor[0]).total_seconds()) if anchor else 0
    out=(RUN/'gaehwasan.out').read_text(errors='replace') if (RUN/'gaehwasan.out').exists() else ''
    complete='FDS completed successfully' in out
    result={'status':'completed' if complete else 'running','checked_at_utc':now.isoformat(),
        'simulation_time_s':current,'duration_s':end,'percent':round(100*current/end,1),
        'elapsed_wall_minutes':round((now-datetime.fromisoformat(manifest['started_at_utc'])).total_seconds()/60,1),
        'recent_sim_seconds_per_wall_second':round(rate,3),
        'estimated_remaining_wall_minutes':round((end-current)/rate/60,1) if rate>0 else None}
    (RUN/'progress.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':progress()
