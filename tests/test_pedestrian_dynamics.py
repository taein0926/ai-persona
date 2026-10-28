import math
import jupedsim as jps
from shapely.geometry import box
from persona_evac.integration.station_dynamics import DT,RADIUS,make_simulation,parameters,plan_overtake

def test_real_jupedsim_passes_slower_person_without_overlap():
    sim = make_simulation(box(0, 0, 30, 4))
    stage = sim.add_waypoint_stage((29, 2), 0.5)
    journey = sim.add_journey(jps.JourneyDescription([stage]))
    slow = sim.add_agent(parameters((7, 2), journey, stage, 0.6))
    fast = sim.add_agent(parameters((2, 2), journey, stage, 1.8))
    min_gap, lateral = 100, 0
    passed = False
    planned = False
    for i in range(400):
        if not planned:
            planned = bool(plan_overtake(sim,box(0,0,30,4),fast,(29,2),stage))
        sim.iterate()
        a, b = sim.agent(slow).position, sim.agent(fast).position
        min_gap = min(min_gap, math.dist(a, b)-2*RADIUS)
        lateral = max(lateral, abs(b[1]-2))
        passed |= b[0] > a[0]+1
    assert passed, "Fast pedestrian must actually get in front of the slow pedestrian"
    assert lateral > RADIUS, "Passing requires lateral motion, not passing through the body"
    assert min_gap >= -0.01

def test_overtaking_rejected_when_corridor_has_no_side_space():
    geom = box(0,0,30,0.7)
    sim = make_simulation(geom)
    stage = sim.add_waypoint_stage((29,0.35),0.3)
    journey = sim.add_journey(jps.JourneyDescription([stage]))
    sim.add_agent(parameters((5,0.35),journey,stage,0.6))
    fast = sim.add_agent(parameters((2,0.35),journey,stage,1.8))
    assert plan_overtake(sim,geom,fast,(29,0.35),stage) is None

def test_small_speed_difference_does_not_trigger_intentional_overtaking():
    geom=box(0,0,30,4);sim=make_simulation(geom)
    stage=sim.add_waypoint_stage((29,2),.3);journey=sim.add_journey(jps.JourneyDescription([stage]))
    front=sim.add_agent(parameters((5,2),journey,stage,1.3))
    actor=sim.add_agent(parameters((2,2),journey,stage,1.6))
    assert plan_overtake(sim,geom,actor,(29,2),stage) is None
    # A fast target speed alone is insufficient; a genuinely slow front walker is required.
    assert plan_overtake(sim,geom,actor,(29,2),stage,observed_speeds={front:1.3}) is None
    assert plan_overtake(sim,geom,actor,(29,2),stage,observed_speeds={front:.6}) is not None
