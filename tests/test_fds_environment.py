"""Coupling invariants and opt-in checks against completed real FDS artifacts."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from persona_evac.integration.fds_environment import FireField, geometry_hash, sha
from fds_fixture import FIRE, config


def field_fixture(tmp_path):
    # Deliberately asymmetric fields detect x/y/floor/time swaps.
    data = {"times": np.array([30, 40]), "x": np.array([0, 60, 120]), "y": np.array([0, 10, 20])}
    for floor, multiplier in [("B1", 1), ("B2", 2)]:
        k = np.array([[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
                      [[2, 3, 4], [5, 6, 7], [8, 9, 10]]], dtype=float)*multiplier
        data[floor+"_k"] = k
        data[floor+"_temperature"] = 20+10*k
    np.savez_compressed(tmp_path / "fields.npz", **data)
    deck = tmp_path / "input.fds"
    deck.write_text("test fixture")
    meta = {"fire": FIRE, "deck_sha256": sha(deck), "geometry_sha256": geometry_hash(config()),
            "fields_sha256": sha(tmp_path / "fields.npz")}
    (tmp_path / "metadata.json").write_text(json.dumps(meta))
    return FireField(config(), tmp_path, deck)


def test_sampling_uses_correct_floor_axes_time_and_rejects_extrapolation(tmp_path):
    field = field_fixture(tmp_path)
    a = field.sample("B1", (60, 0), 5)
    assert a["extinction_per_m"] == 4
    assert a["temperature_c"] == 60
    assert a["speed_factor"] < 1
    assert a["visibility_m_capped_100"] == 0.75
    assert field.sample("B2", (60, 0), 5)["extinction_per_m"] == 8
    with pytest.raises(ValueError, match="FDS"):
        field.sample("B1", (60, 0), 10.1)
    with pytest.raises(ValueError, match="outside"):
        field.sample("B1", (121, 0), 0)


def test_stair_smoke_uses_physical_height_and_temperature_is_declared_interpolation(tmp_path):
    field=field_fixture(tmp_path)
    field.data.update(volume_x=np.array([60]),volume_y=np.array([0]),volume_z=np.array([1.5,5.5]),
                      volume_k=np.array([[[[2.,6.]]],[[[4.,8.]]]]))
    c={'floors':[{'id':'B2','elevation_m':0},{'id':'B1','elevation_m':4}],
       'nodes':{'a':{'floor':'B2','x':60,'y':0},'b':{'floor':'B1','x':60,'y':0}}}
    s={'id':'S','from_node':'a','to_node':'b'}
    segment={'from':c['nodes']['a'],'to':c['nodes']['b']}
    reading=field.sample_stair(c,s,segment,.5,5)
    assert reading['physical_position_m']==pytest.approx([60,0,2])
    assert reading['extinction_per_m']==pytest.approx(5)
    assert reading['visibility_m_capped_100']==pytest.approx(.6)
    assert reading['smoke_sampling']=='height_slices'
    assert reading['temperature_sampling']=='connected_floor_interpolation'
    assert reading['temperature_c']==pytest.approx(80)
    assert reading['speed_factor_applied'] is False


def test_field_provenance_and_missing_data_fail_closed(tmp_path):
    with pytest.raises(ValueError, match="FDS"):
        FireField(config(), tmp_path)
    field_fixture(tmp_path)
    changed = deepcopy(config())
    changed["floors"][0]["walkable"][0][2] = 121
    with pytest.raises(ValueError, match="FDS"):
        FireField(changed, tmp_path, tmp_path / "input.fds")
    (tmp_path / "fields.npz").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="해시"):
        FireField(config(), tmp_path, tmp_path / "input.fds")


