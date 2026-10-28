"""Regression for the four stranded personas at the real F1 obstacle corner."""
import json
import math
from pathlib import Path

import jupedsim as jps
from shapely.geometry import LineString

from persona_evac.integration.station_dynamics import (
    RADIUS,make_simulation,parameters,pedestrian_geometry,plan_cluster_detour,
)
from persona_evac.integration.station_geometry import floor_geometry


def test_compact_counterflow_at_f1_corner_can_back_off_and_exit():
    c=json.loads(Path('configs/scenarios/gaehwasan.json').read_text(encoding='utf8'))
    geom=pedestrian_geometry(floor_geometry(next(f for f in c['floors'] if f['id']=='F1')))
    # Actual recorded positions and goals; three people head to exit 1, one to exit 2.
    records=[
        ((112.6588073882838,29.10994942690381),(118.79625029166667,40.10227108333333)),
        ((112.53750386495835,29.532897952411183),(120.05557005833333,39.766454216666666)),
        ((112.72982555670465,29.92864075636834),(128.00375015,6.19625165)),
        ((112.71418559399274,28.673448276467195),(120.05557005833333,39.766454216666666)),
    ]
    sim=make_simulation(geom);actors=[]
    for start,goal in records:
        stage=sim.add_waypoint_stage(goal,.35)
        journey=sim.add_journey(jps.JourneyDescription([stage]))
        actors.append((sim.add_agent(parameters(start,journey,stage,1)),stage,goal))
    # The old lateral-then-forward planner has no path at this recorded instant.
    actor,stage,goal=actors[1];start=records[1][0]
    route=plan_cluster_detour(sim,geom,actor,goal,stage)
    assert route and math.dist(sim.agent(actor).position,start)<1e-8
    assert sum((route[0][i]-start[i])*(goal[i]-start[i]) for i in (0,1))<0
    assert geom.covers(LineString([start]+route+[goal]).buffer(RADIUS-.005))
    active=dict((a,(stage,goal)) for a,stage,goal in actors)
    for _ in range(2400):
        sim.iterate()
        ids=list(active)
        for i,a in enumerate(ids):
            for b in ids[i+1:]:
                assert math.dist(sim.agent(a).position,sim.agent(b).position)>=2*RADIUS-.01
        for a,(_,goal) in list(active.items()):
            if math.dist(sim.agent(a).position,goal)<.6:
                sim.mark_agent_for_removal(a);del active[a]
        if not active:break
    assert not active
