from copy import deepcopy
import pytest
from pydantic import ValidationError
from persona_evac.integration.station import StationRequest, initialize_station
from persona_evac.integration.stair_dynamics import stair_speed_factor
from station_fixtures import config, source


def test_initialization_keeps_originals_and_does_not_choose_global_routes():
    c = config()
    s = source(100)
    before = deepcopy(s)
    r = initialize_station(s, c, StationRequest(agent_count=100))
    assert s == before
    assert all(not a["segments"] and a["evacuation_time_s"] is None for a in r["agents"])
    for a, original in zip(r["agents"], s["variants"]["C"]):
        assert a["persona"] == original["persona"]
        assert (
            a["pre_movement_s"] == original["pre_movement_s"]
            and a["desired_speed_mps"] == original["desired_speed_mps"]
        )
    assert not any("route_plan" in a or "planned_exit" in a for a in r["agents"])


@pytest.mark.parametrize(
    "field,value",
    [
        ("travel_time_s", 0),
        ("travel_time_s", float("nan")),
        ("entry_rate_per_s", -1),
        ("capacity", 1.5),
    ],
)
def test_invalid_stair_settings_are_rejected(field, value):
    values = {"id": "A_W", "travel_time_s": 12, "entry_rate_per_s": 0.5, "capacity": 4}
    values[field] = value
    with pytest.raises(ValidationError):
        StationRequest(stairs=[values])


def test_stair_width_flow_storage_and_override_provenance():
    c = config()
    before = deepcopy(c)
    s = c["stairs"][0]
    override = {k: s[k] for k in ("id", "width_m", "length_m", "speed_mps", "density_per_m2")}
    override["width_m"] = 0.65
    r = initialize_station(source(), c, StationRequest(stairs=[override]))
    changed = r["scenario"]["stairs"][0]
    assert changed["parallel_lanes"] == 1 and changed["width_basis"] == "assumed"
    assert changed["travel_time_s"] == pytest.approx(changed["length_m"] / changed["speed_mps"])
    assert changed["entry_rate_per_s"] == pytest.approx(
        changed["density_per_m2"] * changed["speed_mps"] * changed["width_m"]
    )
    assert c == before
    assert stair_speed_factor(changed, 0) == 1 and stair_speed_factor(
        changed, changed["capacity"]
    ) == pytest.approx(0.55)


def test_unknown_duplicate_and_insufficient_space_are_rejected():
    c = config()
    values = {"id": "missing", "travel_time_s": 12, "entry_rate_per_s": 0.5, "capacity": 4}
    with pytest.raises(ValueError):
        initialize_station(source(), c, StationRequest(stairs=[values]))
    values["id"] = "A_W"
    with pytest.raises(ValueError):
        initialize_station(source(), c, StationRequest(stairs=[values, values]))
    override = {"id": "A_W", "width_m": 0.3, "length_m": 1, "speed_mps": 0.5, "density_per_m2": 0.1}
    with pytest.raises(ValueError):
        initialize_station(source(), c, StationRequest(stairs=[override]))
