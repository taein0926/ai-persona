from __future__ import annotations

import time
import os
import re
from pathlib import Path
from typing import Protocol

from openai import OpenAI
from dotenv import load_dotenv

from persona_evac.hashing import canonical_json, sha256_value, stable_seed
from persona_evac.schemas import (
    BehaviorMapping,
    BehaviorReasoning,
    PersonaProfile,
    Tendency,
)


load_dotenv()


PROMPT_VERSION = "1"
SCHEMA_VERSION = "1"
SYSTEM_PROMPT = """당신은 합성 Persona를 대피 행동의 정성적 범주로만 매핑한다.
응답은 제공된 스키마를 정확히 따르고 물리량, 출구 번호, 확률을 만들지 않는다.
Persona는 실제 재난 행동의 ground truth가 아니므로 근거를 짧고 비단정적으로 작성한다."""
GROQ_MAPPING_PROMPT = SYSTEM_PROMPT + """
각 reasoning_summary는 한국어 30자 이내로 입력의 근거와 추정만 적는다.
선택한 빠름·보통·느림 범주와 근거를 일치시킨다. 입력에 없는 건강·경험은 만들지 않는다.
성향을 구분할 근거가 부족하면 normal로 두고 '근거 부족으로 보통 가정'이라고 적는다."""


class MappingProvider(Protocol):
    source_name: str

    def map(self, profile: PersonaProfile, scenario_context: dict[str, object]) -> BehaviorMapping: ...


class JsonCache:
    def __init__(self, directory: Path):
        self.directory = directory

    def key(
        self,
        *,
        profile: PersonaProfile,
        scenario_context: dict[str, object],
        provider: str,
        model: str,
        settings: dict[str, object],
    ) -> str:
        return sha256_value(
            {
                "profile": profile.model_dump(mode="json"),
                "scenario_context": scenario_context,
                "provider": provider,
                "model": model,
                "settings": settings,
                "prompt_version": PROMPT_VERSION,
                "schema_version": SCHEMA_VERSION,
            }
        )

    def get(self, key: str) -> BehaviorMapping | None:
        path = self.directory / f"{key}.json"
        if not path.exists():
            return None
        return BehaviorMapping.model_validate_json(path.read_text(encoding="utf-8"))

    def put(self, key: str, mapping: BehaviorMapping) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"{key}.json"
        path.write_text(mapping.model_dump_json(indent=2), encoding="utf-8")


class MockProvider:
    source_name = "mock"
    model = "deterministic-hash-v1"

    def map(self, profile: PersonaProfile, scenario_context: dict[str, object]) -> BehaviorMapping:
        tendencies = [Tendency.FAST, Tendency.NORMAL, Tendency.SLOW]
        seed = stable_seed(profile.uuid, scenario_context, PROMPT_VERSION)
        return BehaviorMapping(
            response_tendency=tendencies[seed % 3],
            speed_tendency=tendencies[(seed >> 3) % 3],
            reasoning_summary=BehaviorReasoning(
                response_tendency="고정 해시 기반 테스트 범주이며 실제 행동 예측이 아닙니다.",
                speed_tendency="고정 해시 기반 테스트 범주이며 실제 행동 예측이 아닙니다.",
            ),
        )


class OpenAIProvider:
    source_name = "openai"

    def __init__(
        self,
        *,
        model: str | None = None,
        cache_dir: Path = Path("data/cache/llm"),
        max_retries: int = 3,
        client: OpenAI | None = None,
    ):
        self.model = model or os.environ.get("OPENAI_MODEL", "")
        if not self.model:
            raise ValueError("OPENAI_MODEL environment variable is required for the OpenAI provider")
        self.cache = JsonCache(cache_dir)
        self.max_retries = max_retries
        self.client = client or OpenAI()
        self.settings = {"reasoning_effort": "low", "store": False, "tools": []}

    def map(self, profile: PersonaProfile, scenario_context: dict[str, object]) -> BehaviorMapping:
        key = self.cache.key(
            profile=profile,
            scenario_context=scenario_context,
            provider=self.source_name,
            model=self.model,
            settings=self.settings,
        )
        cached = self.cache.get(key)
        if cached:
            return cached
        user_payload = canonical_json(
            {"persona": profile.model_dump(mode="json"), "scenario": scenario_context}
        )
        for attempt in range(self.max_retries):
            try:
                response = self.client.responses.parse(
                    model=self.model,
                    reasoning={"effort": "low"},
                    store=False,
                    tools=[],
                    input=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_payload},
                    ],
                    text_format=BehaviorMapping,
                )
                if getattr(response, "status", "completed") != "completed":
                    raise RuntimeError(f"OpenAI response incomplete: {response.status}")
                mapping = response.output_parsed
                if mapping is None:
                    raise RuntimeError("OpenAI response was refused or contained no parsed output")
                validated = BehaviorMapping.model_validate(mapping)
                self.cache.put(key, validated)
                return validated
            except Exception as exc:
                status = getattr(exc, "status_code", None)
                transient = status in {408, 429, 500, 502, 503, 504}
                if not transient or attempt == self.max_retries - 1:
                    raise
                time.sleep(2**attempt)
        raise RuntimeError("OpenAI mapping exhausted retries")


class GroqProvider:
    source_name = "groq"
    base_url = "https://api.groq.com/openai/v1"

    def __init__(
        self,
        *,
        model: str | None = None,
        cache_dir: Path = Path("data/cache/llm"),
        max_retries: int = 3,
        client: OpenAI | None = None,
    ):
        self.model = model or os.environ.get("GROQ_MODEL", "")
        if not self.model:
            raise ValueError("GROQ_MODEL environment variable is required for the Groq provider")
        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key and client is None:
            raise ValueError("GROQ_API_KEY is not configured. Add it to the project root .env file.")
        self.cache = JsonCache(cache_dir)
        self.max_retries = max_retries
        self.client = client or OpenAI(api_key=api_key, base_url=self.base_url, timeout=45, max_retries=0)
        self.settings = {"response_format": "BehaviorMapping", "base_url": self.base_url,
                         "max_completion_tokens": 384, "system_prompt": GROQ_MAPPING_PROMPT}

    def map(self, profile: PersonaProfile, scenario_context: dict[str, object]) -> BehaviorMapping:
        key = self.cache.key(
            profile=profile,
            scenario_context=scenario_context,
            provider=self.source_name,
            model=self.model,
            settings=self.settings,
        )
        cached = self.cache.get(key)
        if cached:
            return cached
        user_payload = canonical_json(
            {"persona": profile.model_dump(mode="json"), "scenario": scenario_context}
        )
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.parse(
                    model=self.model,
                    temperature=0,
                    max_completion_tokens=384,
                    messages=[
                        {"role": "system", "content": GROQ_MAPPING_PROMPT},
                        {"role": "user", "content": user_payload},
                    ],
                    response_format=BehaviorMapping,
                )
                choice = response.choices[0]
                if getattr(choice, "finish_reason", "stop") == "length":
                    raise RuntimeError("Groq response was truncated")
                mapping = getattr(choice.message, "parsed", None)
                if mapping is None:
                    raise RuntimeError("Groq response contained no parsed output")
                validated = BehaviorMapping.model_validate(mapping)
                self.cache.put(key, validated)
                return validated
            except Exception as exc:
                status = getattr(exc, "status_code", None)
                transient = status in {408, 429, 500, 502, 503, 504}
                if not transient or attempt == self.max_retries - 1:
                    raise
                retry_after = getattr(getattr(exc, "response", None), "headers", {}).get("retry-after", "10")
                limit = re.search(r"on ((?:input |output )?tokens per (?:minute|day) \([A-Z]+\)): Limit ([\d,]+), Used ([\d.,]+), Requested ([\d,]+)", str(exc))
                if limit:
                    print(f"Groq quota: {limit.group(0)}", flush=True)
                    if "per day" in limit.group(1):
                        raise RuntimeError("Groq daily token quota exhausted; cached mappings are preserved") from None
                if status == 429 and float(retry_after) > 300:
                    raise RuntimeError(f"Groq quota requires waiting {retry_after}s; cached mappings are preserved") from None
                delay = min(60, max(10, float(retry_after))) if status == 429 else 2**attempt
                print(f"Groq mapping retry: HTTP {status}, waiting {delay}s", flush=True)
                time.sleep(delay)
        raise RuntimeError("Groq mapping exhausted retries")


def fallback_mapping(profile: PersonaProfile, scenario_context: dict[str, object]) -> BehaviorMapping:
    mapping = MockProvider().map(profile, scenario_context)
    return mapping.model_copy(
        update={
            "reasoning_summary": BehaviorReasoning(
                response_tendency="OpenAI 실패 후 사용한 결정론적 데모 fallback입니다.",
                speed_tendency="OpenAI 실패 후 사용한 결정론적 데모 fallback입니다.",
            )
        }
    )
