from __future__ import annotations

import json
from pathlib import Path

from persona_evac.integration.pyfds_adapter import build_persona_scenario
from persona_evac.integration.runner import run_experiment
from persona_evac.schemas import ExperimentResults


def test_adapter_builds_ten_one_agent_distributions(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    config = json.loads(Path("configs/experiments/mvp.json").read_text(encoding="utf-8"))
    config["output_dir"] = str(tmp_path)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setenv("PERSONA_EVAC_OUTPUT", str(tmp_path))
    result_path = run_experiment(config_path)
    result = ExperimentResults.model_validate_json(result_path.read_text(encoding="utf-8"))
    output = tmp_path / "derived.json"
    build_persona_scenario(
        upstream_config=Path("vendor/pyFDS-Evac/assets/t_junction/config.json"),
        agents=result.variants["C"],
        output=output,
        seed=1,
    )
    raw = json.loads(output.read_text(encoding="utf-8"))
    assert (tmp_path / "geometry.wkt").read_text(encoding="utf-8")
    assert len(raw["distributions"]) == 10
    assert all(key.startswith("jps-distributions_") for key in raw["distributions"])
    assert len({json.dumps(item["coordinates"]) for item in raw["distributions"].values()}) == 10
    assert sum(item["parameters"]["number"] for item in raw["distributions"].values()) == 10
    for item in raw["distributions"].values():
        params = item["parameters"]
        assert params["use_flow_spawning"] is False
        assert params["use_premovement"] is True
        assert params["premovement_param_a"] == params["premovement_param_b"]
        assert params["familiarity"] == "discovery"
        assert params["radius"] == 0.15
