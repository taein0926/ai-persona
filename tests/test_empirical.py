from __future__ import annotations

from pathlib import Path

from persona_evac.empirical.julich_parser import parse_pair_file, trajectory_features
from persona_evac.empirical.resolver import EmpiricalResolver, load_empirical_records
from persona_evac.schemas import SocialMode, Tendency


FIXTURE = Path("data/fixtures/julich_features.csv")


def test_joint_resolver_is_deterministic_and_preserves_requested_bin() -> None:
    resolver = EmpiricalResolver(load_empirical_records(FIXTURE), master_seed=7)
    first = resolver.resolve(
        persona_uuid="p-1",
        variant="C",
        treatment=SocialMode.GROUP,
        response=Tendency.NORMAL,
        speed=Tendency.FAST,
    )
    second = resolver.resolve(
        persona_uuid="p-1",
        variant="C",
        treatment=SocialMode.GROUP,
        response=Tendency.NORMAL,
        speed=Tendency.FAST,
    )
    assert first == second
    assert first.response_treatment == SocialMode.GROUP
    assert first.resolved_response_tendency == Tendency.NORMAL
    assert first.resolved_speed_tendency == Tendency.FAST
    assert first.observed_speed_proxy_mps == first.desired_speed_mps
    assert first.speed_is_proxy is True


def test_julich_pair_parser_and_feature_extraction(tmp_path: Path) -> None:
    pair = tmp_path / "1_individual.txt"
    pair.write_text("0 0 0 0.1 0.2 0.3\n0 0 0 0 0 0\n", encoding="utf-8")
    trajectories = parse_pair_file(pair)
    assert len(trajectories) == 1
    response, speed = trajectory_features(
        trajectories[0],
        metres_per_pixel=1.0,
        experiment_start_frame=1,
        exit_midpoint_px=(0.3, 0.0),
        exit_zone_m=0.01,
    )
    assert response == 0.125
    assert speed == 1.6
