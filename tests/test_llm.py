from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from persona_evac.llm.providers import GroqProvider, MockProvider, OpenAIProvider
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


def test_openai_cache_prevents_second_call(tmp_path) -> None:
    mapping = BehaviorMapping(
        response_tendency="normal",
        speed_tendency="fast",
        reasoning_summary={
            "response_tendency": "fixture response",
            "speed_tendency": "fixture speed",
        },
    )

    class Responses:
        calls = 0

        def parse(self, **kwargs):
            self.calls += 1
            return SimpleNamespace(status="completed", output_parsed=mapping)

    client = SimpleNamespace(responses=Responses())
    provider = OpenAIProvider(client=client, cache_dir=tmp_path, model="test-model")
    profile = PersonaProfile(uuid="p", persona="fixture", age=35)
    assert provider.map(profile, {"scenario": "test"}) == mapping
    assert provider.map(profile, {"scenario": "test"}) == mapping
    assert client.responses.calls == 1


def test_groq_cache_and_structured_mapping(tmp_path) -> None:
    mapping = BehaviorMapping(
        response_tendency="slow",
        speed_tendency="normal",
        reasoning_summary={
            "response_tendency": "fixture response",
            "speed_tendency": "fixture speed",
        },
    )

    class Completions:
        calls = 0

        def parse(self, **kwargs):
            self.calls += 1
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(parsed=mapping), finish_reason="stop")]
            )

    client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    provider = GroqProvider(client=client, cache_dir=tmp_path, model="qwen/test")
    profile = PersonaProfile(uuid="p", persona="fixture", age=35)
    assert provider.map(profile, {"scenario": "test"}) == mapping
    assert provider.map(profile, {"scenario": "test"}) == mapping
    assert client.chat.completions.calls == 1


def test_groq_requires_api_key(monkeypatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("GROQ_MODEL", "qwen/test")
    with pytest.raises(ValueError, match="GROQ_API_KEY is not configured"):
        GroqProvider()


def test_groq_daily_quota_stops_without_repeated_requests(tmp_path, monkeypatch):
    import httpx
    from openai import RateLimitError
    class Completions:
        calls = 0
        def parse(self, **kwargs):
            self.calls += 1
            response = httpx.Response(429, request=httpx.Request("POST", "https://example.invalid"))
            raise RateLimitError("Rate limit on tokens per day (TPD): Limit 200000, Used 199900, Requested 1000",
                                 response=response, body=None)
    client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    provider = GroqProvider(client=client, cache_dir=tmp_path, model="qwen/test")
    monkeypatch.setattr("persona_evac.llm.providers.time.sleep", lambda _: pytest.fail("Daily quota must not retry"))
    with pytest.raises(RuntimeError, match="daily token quota"):
        provider.map(PersonaProfile(uuid="p", age=30), {})
    assert client.chat.completions.calls == 1
