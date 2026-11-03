"""실제 FDS 균일 CO 검증실의 농도·장치 FED와 새 표본 추출·적분을 비교한다."""
import argparse
import csv
import json
from pathlib import Path

from persona_evac.integration.fds_environment import FireField,export_fds
from persona_evac.integration.gas_exposure import fed_rate_per_s

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'output/qa/gas_20261102/coupling'


def prepare():
    RUN.mkdir(parents=True,exist_ok=True)
    original=ROOT/'vendor/pyFDS-Evac/assets/fed_incap_co_4000ppm/fed_incap_co_4000ppm.fds'
    text=original.read_text(encoding='utf8').replace("CHID='demo_homogeneous_CO_4000ppm'","CHID='gaehwasan'")
    text=text.replace('T_END=420','T_END=2')
    text=text.replace('&TIME T_END=2 /','&TIME T_END=2 /\n&DUMP DT_SLCF=0.5, DT_DEVC=0.05 /')
    text=text.replace('&TAIL /',"&SLCF PBZ=2.0, QUANTITY='TEMPERATURE' /\n&TAIL /")
    (RUN/'gaehwasan.fds').write_text(text,encoding='utf8')
    c={'name':'균일 CO 검증실','station_slug':'co_coupling',
       'floors':[{'id':'B2','elevation_m':.5}], 'nodes':{},
       'fire':{'slice_height_m':{'B2':2.},'volume_slice_heights_m':[2.],
               'mesh_cell_m':.5,'time_end_s':2.,'evacuation_start_s':0}}
    (RUN/'scenario.json').write_text(json.dumps(c),encoding='utf8')


def verify():
    deck=RUN/'gaehwasan.fds';config=RUN/'scenario.json'
    export_fds(RUN,RUN/'derived',config,deck)
    field=FireField(json.loads(config.read_bytes()),RUN/'derived',deck);field.require_gases()
    reading=field.sample_gases_xyz([14.9,14.9,.5],1.)
    assert abs(reading['co_ppm']-4000)<2,reading
    assert abs(reading['co2_percent']-.05)<.0001,reading
    assert abs(reading['o2_percent']-20.9)<.01,reading
    with (RUN/'gaehwasan_devc.csv').open(encoding='utf8',newline='') as file:
        rows=list(csv.reader(file))
    column=[s.strip() for s in rows[1]].index('FED_center')
    expected=float(rows[-1][column]);dose=0.
    for tick in range(40):
        reading=field.sample_gases_xyz([14.9,14.9,.5],(tick+.5)*.05)
        dose+=fed_rate_per_s(reading)*.05
    assert abs(dose-expected)<max(1e-6,expected*.005),(dose,expected)
    report={'status':'통과','concentrations':reading,'sampled_fed_at_2s':dose,
            'fds_device_fed_at_2s':expected,'relative_error':abs(dose-expected)/expected}
    (RUN/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description='FDS와 FED 적분을 대조 검증합니다.')
    parser.add_argument('--prepare',action='store_true')
    if parser.parse_args().prepare:prepare()
    else:verify()
