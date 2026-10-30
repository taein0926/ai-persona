"""최종 검증기가 새 성향 버전과 실제 연결 통행의 오류를 잡는지 검사한다."""
import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def native_audit(monkeypatch):
    spec=importlib.util.spec_from_file_location('final_result_audit',Path('scripts/audit_private_result.py'))
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from persona_evac.integration import station_connectors
    connector={'kind':'fare_gate','parallel_lanes':1,'capacity':1,
               'native_lane_specs':[{'entry':[0,0,4],'destination':[2,0,4]}]}
    monkeypatch.setattr(station_connectors,'compile_network',lambda _: {'connectors':{'G_01':connector}})
    return module.audit_native_passages


def passage(aid,start,end):
    return {'agent_id':aid,'stair_id':'G_01','engine':'jupedsim','kind':'fare_gate','lane':0,
            'arrival':start,'entry':start,'exit':end,
            'native_motion':[[start,0,0,4],[start+.5,1,0,4],[end,2,0,4]]}


def test_gate_can_transfer_ownership_at_exact_exit(native_audit):
    result={'summary':{'total_time_s':10},'passages':[passage(1,0,1),passage(2,1,2)]}
    report=native_audit(result,{})
    assert report['peak_occupancy']=={'G_01':1}
    assert report['native_motion_samples']==6


def test_gate_double_ownership_is_rejected(native_audit):
    result={'summary':{'total_time_s':10},'passages':[passage(1,0,2),passage(2,1,3)]}
    with pytest.raises(AssertionError):native_audit(result,{})


def test_incorrect_native_exit_position_is_rejected(native_audit):
    p=passage(1,0,1);p['native_motion'][-1][1]=20
    with pytest.raises(AssertionError):native_audit({'summary':{'total_time_s':10},'passages':[p]}, {})


def test_unfinished_gate_passage_keeps_ownership(native_audit):
    first=passage(1,0,1);first['exit']=None
    result={'summary':{'total_time_s':10},'passages':[first,passage(2,2,3)]}
    with pytest.raises(AssertionError):native_audit(result,{})
