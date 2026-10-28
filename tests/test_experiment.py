from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq

from persona_evac.integration.runner import run_experiment
from persona_evac.schemas import ExperimentResults


def test_mvp_experiment_has_three_paired_variants(tmp_path: Path, monkeypatch) -> None:
    config = json.loads(Path("configs/experiments/mvp.json").read_text(encoding="utf-8"))
    config["output_dir"] = str(tmp_path)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setenv("PERSONA_EVAC_OUTPUT", str(tmp_path))
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    result_path = run_experiment(config_path)
    result = ExperimentResults.model_validate_json(result_path.read_text(encoding="utf-8"))
    assert set(result.variants) == {"A", "B", "C"}
    assert all(len(result.variants[key]) == 10 for key in result.variants)
    mappings = [
        [(agent.agent_slot, agent.persona.uuid) for agent in result.variants[key]]
        for key in ("A", "B", "C")
    ]
    assert mappings[0] == mappings[1] == mappings[2]
    assert len({uuid for _, uuid in mappings[0]}) == 10
    assert all(summary.run_success for summary in result.summaries.values())
    assert all(summary.all_evacuated for summary in result.summaries.values())
    assert result.backend == "deterministic_mvp"
    assert "FDS/JuPedSim" in result.backend_notice
    runtime = pq.read_table(result_path.parent / "agent_runtime_map.parquet").to_pylist()
    assert len(runtime) == 30
    assert len({(row["variant"], row["jupedsim_agent_id"]) for row in runtime}) == 30
    assert (tmp_path / "latest/results.json").exists()
    assert all(agent.scenario_context.social_mode == "individual" for agent in result.variants["C"])
    assert all(agent.scenario_context.familiarity == "discovery" for agent in result.variants["C"])
    assert all(agent.mapping is not None for agent in result.variants["C"])
    assert all(not hasattr(agent.mapping, "social_mode") for agent in result.variants["C"])


def test_frontend_contains_no_random_data_generation() -> None:
    source = Path("app.js").read_text(encoding="utf-8")
    assert "Math.random" not in source
    assert "/api/results" in source
