from __future__ import annotations

import pytest
from pydantic import ValidationError

from persona_evac.llm import MockProvider
from persona_evac.schemas import BehaviorMapping, PersonaProfile


def test_schema_rejects_extra_or_numeric_fields() -> None:
    with pytest.raises(ValidationError):
        BehaviorMapping.model_validate(
            {
                "response_tendency": "fast",
                "speed_tendency": 1.2,
                "reasoning_summary": {
                    "response_tendency": "invalid",
                    "speed_tendency": "invalid",
                },
                "exit_id": "exit_1",
            }
        )


def test_mock_provider_is_deterministic() -> None:
    profile = PersonaProfile(uuid="p", persona="fixture", age=40)
    provider = MockProvider()
    assert provider.map(profile, {"scenario": "x"}) == provider.map(profile, {"scenario": "x"})


def test_common_prompt_and_cache_key_preserve_existing_ollama_artifacts(tmp_path):
    import hashlib
    from persona_evac.llm.common import MAPPING_PROMPT, JsonCache
    assert hashlib.sha256(MAPPING_PROMPT.encode()).hexdigest() == "c07371d49d019f1b17979cbf4aad2db0533b1685842fa6f0aab93ee6f6772284"
    cache_digest = JsonCache(tmp_path).key(
        profile=PersonaProfile(uuid="p", age=35), scenario_context={"scenario": "test"},
        provider="ollama", model="qwen3:4b", settings={"system_prompt": MAPPING_PROMPT},
    )
    assert cache_digest == "c29ff2168f8b3e88a15153cfcfd6c2be0df364944fc946d7dc3ca1ba99841982"
