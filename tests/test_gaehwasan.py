import json
import math
from pathlib import Path

import pytest
from shapely.geometry import Point

from persona_evac.integration.station import StationRequest, simulate_station
from persona_evac.integration.station_geometry import floor_geometry
from station_fixtures import source


def station():
    return json.loads(Path("configs/scenarios/gaehwasan.json").read_text(encoding="utf-8"))


def test_drawing_calibration_width_evidence_and_body_clear_spawn():
    c=station()
    scale=c["drawing"]["calibration"]["m_per_pdf_point"]
    assert scale*15.12 == pytest.approx(4)
    assert abs(scale*18.96-5.028)<.02
    assert {f["id"] for f in c["floors"]}=={"B2","B1","F1"}
    for s in c["stairs"]:
        e=s["width_reference"]
        assert s["width_m"] == pytest.approx(round(math.dist(*e["pdf_endpoints"])*scale-.2,2))
    f=c["floors"][0]
    for spawn in c["spawn_points"]:
        region=0 if spawn["area"]=="A" else 1
        assert floor_geometry(f,region).buffer(-.24).covers(Point(spawn["x"],spawn["y"]))
    # The center rails are genuinely nonwalkable and the platforms are curved polygons.
    assert not floor_geometry(f).covers(Point(50,20))
    assert len(f["polygons"][0])>15


def test_three_levels_conserve_personas_and_respect_metric_geometry():
    c=station()
    r=simulate_station(source(30),c,StationRequest(agent_count=30,duration_s=5))
    assert len({a["persona"]["uuid"] for a in r["agents"]})==30
    assert r["movement_diagnostics"]["overlap_pair_steps"]==0
    geoms={f["id"]:floor_geometry(f) for f in c["floors"]}
    for a in r["agents"]:
        assert {s["from"]["floor"] for s in a["segments"] if s["status"]=="walking"}<={"B2","B1","F1"}
        for seg in a["segments"]:
            if seg["status"]=="on_stairs": continue
            for p in (seg["from"],seg["to"]):
                assert geoms[p["floor"]].covers(Point(p["x"],p["y"]))
        for prev,nxt in zip(a["segments"],a["segments"][1:]):
            assert prev["end"] == pytest.approx(nxt["start"])
            assert prev["to"] == nxt["from"]
    for s in r["scenario"]["stairs"]:
        assert s["width_basis"]=="drawing_estimate"
        assert 0<s["active_entry_lanes"]<=s["parallel_lanes"]
        assert len(s['physical_valid_lanes'])==s['active_entry_lanes']
        assert set(s['physical_valid_lanes'])<=set(s['entry_lane_numbers'])
        dest=r['scenario']['nodes'][s['to_node']]
        if dest['floor']!='OUTSIDE':
            for lane in s['physical_valid_lanes']:
                offset=(lane-(s['parallel_lanes']-1)/2)*s['width_m']/s['parallel_lanes']
                x=dest['x']+offset*s['portal_tangent'][0];y=dest['y']+offset*s['portal_tangent'][1]
                assert geoms[dest['floor']].buffer(-.24).covers(Point(x,y))
        ps=sorted((p for p in r["passages"] if p["stair_id"]==s["id"]),key=lambda p:p["entry"])
        for a,b in zip(ps,ps[1:]):
            assert b["entry"]-a["entry"] >= 1/s["entry_rate_per_s"]-1e-8


def test_narrow_override_reduces_entry_lanes_and_marks_assumption():
    c=station()
    s=c["stairs"][0]
    override={k:s[k] for k in ("id","width_m","length_m","speed_mps","density_per_m2")}
    override["width_m"]=.65
    r=simulate_station(source(),c,StationRequest(stairs=[override],duration_s=5))
    changed=next(s for s in r["scenario"]["stairs"] if s["id"]==override["id"])
    assert changed["active_entry_lanes"]==1
    assert changed["width_basis"]=="assumed"
    assert len(r['agents'])==10


@pytest.mark.skipif(not Path("output/fds/gaehwasan/derived/metadata.json").exists(),reason="개화산역 FDS 사전 계산 필요")
def test_real_gaehwasan_fds_all_three_levels_and_source_match():
    from persona_evac.integration.fds_environment import FireField
    field=FireField(station())
    assert set(field.meta["fire"]["slice_height_m"])=={"B2","B1","F1"}
    assert field.times[-1]>=479.9
    assert field.data["F1_k"].max()>0
    assert field.data["F1_temperature"].max()>20
    r=simulate_station(source(),station(),StationRequest(fire_mode="fds",duration_s=5))
    assert len(r['agents'])==10
    assert r["fire"]["enabled"]
    assert all(h["source"]=="fds" for h in r["hazard_history"])
