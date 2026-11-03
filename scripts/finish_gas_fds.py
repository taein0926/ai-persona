"""완료된 가스 FDS를 검증·변환하고, 요청 시 과거 결과를 보존하며 활성화한다."""
import argparse
import json
import shutil
from pathlib import Path

from persona_evac.integration.fds_environment import FireField,export_fds,sha
from persona_evac.integration.station_connectors import compile_network,PreparedPath

ROOT=Path(__file__).resolve().parents[1]
FDS_ROOT=ROOT/'output/fds'
RUN=FDS_ROOT/'gaehwasan_20261102_gas'


def finish(activate=False,reuse_fields=False):
    config_path=ROOT/'configs/scenarios/gaehwasan.json'
    if config_path.read_bytes()!=(RUN/'scenario.json').read_bytes():
        raise ValueError('FDS 계산 중 지도·화재 설정이 변경되었습니다.')
    c=json.loads(config_path.read_bytes());deck=RUN/'gaehwasan.fds'
    if not reuse_fields:export_fds(RUN,RUN/'derived',RUN/'scenario.json',deck)
    field=FireField(c,RUN/'derived',deck);field.require_gases()
    if reuse_fields:
        if (field.meta.get('out_sha256')!=sha(RUN/'gaehwasan.out')
                or not field.meta.get('raw_files_sha256')
                or any(sha(RUN/name)!=digest for name,digest in field.meta['raw_files_sha256'].items())):
            raise ValueError('변환 후 FDS 원본이 변경되어 농도장을 재사용할 수 없습니다.')
        field.meta['gas_sampling']['z']='linear between air levels; air-side sample at solid boundaries'
        (RUN/'derived/metadata.json').write_text(json.dumps(field.meta,indent=2),encoding='utf8')
    if field.times[-1]<c['fire']['time_end_s']-.1:
        raise ValueError('FDS 가스 계산 시간이 부족합니다.')
    # Validate every spawn's breathing position throughout the complete run.
    for p in c['spawn_points']:
        for t in (0,c['fire']['time_end_s']/2,c['fire']['time_end_s']):
            field.sample_gases_xyz([p['x'],p['y'],0.],t-field.offset)
    network=compile_network(c);path_samples=0
    for sid,connector in network['connectors'].items():
        for lane in connector['native_lane_specs']:
            path=PreparedPath(connector['native_paths'][lane['branch']])
            steps=max(1,int(path.length/.25)+1)
            for step in range(steps+1):
                xyz=path.position(path.length*step/steps,lane['offset'])
                try:field.sample_gases_xyz(xyz,0.)
                except ValueError as exc:raise ValueError(f'{sid} 연결의 호흡 위치 {xyz}: {exc}') from exc
                path_samples+=1
    report={'status':'validated','deck_sha256':sha(deck),
            'fields_sha256':field.meta['fields_sha256'],
            'source_config_sha256':sha(config_path),'time_end_s':float(field.times[-1]),
            'gas_max_volume_fraction':{key:float(field.data['gas_'+key].max()) for key in ('co','co2','o2')},
            'validated_connector_breathing_positions':path_samples,
            'exported_fields_reused':reuse_fields,
            'activated':False}
    (RUN/'gas_validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    if activate:
        active=FDS_ROOT/'gaehwasan';archive=FDS_ROOT/'gaehwasan_before_20261102_gas'
        for path in (RUN,active,archive):
            if path.resolve().parent!=FDS_ROOT.resolve():
                raise ValueError('FDS 활성화 경로가 지정된 작업 폴더를 벗어났습니다.')
        if archive.exists():raise ValueError('기존 FDS 보관 폴더를 덮어쓸 수 없습니다.')
        current_deck=ROOT/'configs/fds/gaehwasan.fds'
        previous_deck=current_deck.read_bytes()
        new_deck=current_deck.with_suffix('.gas.tmp')
        # Prove the configuration directory is writable before moving results.
        shutil.copyfile(RUN/'gaehwasan.fds',new_deck)
        if active.exists():active.rename(archive)
        try:
            RUN.rename(active)
            new_deck.replace(current_deck)
        except BaseException:
            if active.exists() and not RUN.exists():active.rename(RUN)
            if archive.exists() and not active.exists():archive.rename(active)
            if current_deck.read_bytes()!=previous_deck:current_deck.write_bytes(previous_deck)
            raise
        # fdsreader pickles embed absolute paths from the staging directory.
        cache=active/'gaehwasan.pickle'
        if cache.exists():cache.rename(active/'reader_before_activation.pickle')
        report['activated']=True
        (active/'gas_validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description='가스 농도장을 검증·변환합니다.')
    parser.add_argument('--activate',action='store_true',help='검증 후 기존 FDS를 보존하고 새 결과를 연결합니다.')
    parser.add_argument('--reuse-fields',action='store_true',help='원본·입력·변환 해시가 일치할 때 이미 변환한 농도장을 재사용합니다.')
    args=parser.parse_args()
    finish(args.activate,args.reuse_fields)
