"""고정 합성 자료와 회귀 테스트용 매핑. 모델 추론을 호출하지 않는다."""
from persona_evac.hashing import stable_seed
from persona_evac.llm.common import PROMPT_VERSION
from persona_evac.schemas import BehaviorMapping, BehaviorReasoning, PersonaProfile, Tendency


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
