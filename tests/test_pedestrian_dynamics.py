"""의도적 추월 제거 후에도 정체 해소용 국소 우회와 충돌 제약을 유지한다."""
import math
import jupedsim as jps
from shapely.geometry import box
from persona_evac.integration.station_dynamics import RADIUS,make_simulation,parameters,plan_yield


def test_local_recovery_can_pass_a_blocking_person_without_overlap():
    geom=box(0,0,30,4);sim=make_simulation(geom)
    stage=sim.add_waypoint_stage((29,2),.5)
    journey=sim.add_journey(jps.JourneyDescription([stage]))
    blocker=sim.add_agent(parameters((3,2),journey,stage,0.))
    actor=sim.add_agent(parameters((2,2),journey,stage,1.8))
    assert plan_yield(sim,geom,actor,(29,2),stage)
    minimum,lateral,passed=100,0,False
    for _ in range(400):
        sim.iterate()
        front,pos=sim.agent(blocker).position,sim.agent(actor).position
        minimum=min(minimum,math.dist(front,pos)-2*RADIUS)
        lateral=max(lateral,abs(pos[1]-2))
        passed |= pos[0]>front[0]+1
    assert passed and lateral>RADIUS
    assert minimum>=-.01


def test_local_recovery_rejected_when_corridor_has_no_side_space():
    geom=box(0,0,30,.7);sim=make_simulation(geom)
    stage=sim.add_waypoint_stage((29,.35),.3)
    journey=sim.add_journey(jps.JourneyDescription([stage]))
    sim.add_agent(parameters((3,.35),journey,stage,0.))
    actor=sim.add_agent(parameters((2,.35),journey,stage,1.8))
    assert plan_yield(sim,geom,actor,(29,.35),stage) is None


def test_people_outside_the_blocking_range_do_not_trigger_local_recovery():
    geom=box(0,0,30,4);sim=make_simulation(geom)
    stage=sim.add_waypoint_stage((29,2),.3)
    journey=sim.add_journey(jps.JourneyDescription([stage]))
    sim.add_agent(parameters((5,2),journey,stage,1.3))
    actor=sim.add_agent(parameters((2,2),journey,stage,1.6))
    assert plan_yield(sim,geom,actor,(29,2),stage) is None
