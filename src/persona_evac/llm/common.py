"""성향 매핑의 공통 프롬프트와 기존 응답 캐시 형식."""
from pathlib import Path
from typing import Protocol

from persona_evac.hashing import sha256_value
from persona_evac.schemas import BehaviorMapping, PersonaProfile


PROMPT_VERSION = "1"
SCHEMA_VERSION = "1"
SYSTEM_PROMPT = """당신은 합성 Persona를 대피 행동의 정성적 범주로만 매핑한다.
응답은 제공된 스키마를 정확히 따르고 물리량, 출구 번호, 확률을 만들지 않는다.
Persona는 실제 재난 행동의 ground truth가 아니므로 근거를 짧고 비단정적으로 작성한다."""
MAPPING_PROMPT = SYSTEM_PROMPT + """
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
