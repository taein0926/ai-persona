from __future__ import annotations

import csv
import math
import random
from pathlib import Path

from persona_evac.hashing import stable_seed
from persona_evac.schemas import EmpiricalRecord, ResolvedBehavior, SocialMode, Tendency


RESOLVER_VERSION = "1"


def load_empirical_records(path: Path) -> list[EmpiricalRecord]:
    with path.open(encoding="utf-8", newline="") as handle:
        records = [EmpiricalRecord.model_validate(row) for row in csv.DictReader(handle)]
    if len({record.record_id for record in records}) != len(records):
        raise ValueError("Empirical data contains duplicate record IDs")
    return categorize(records)


def _quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _response_category(value: float, q33: float, q67: float) -> Tendency:
    if value <= q33:
        return Tendency.FAST
    if value <= q67:
        return Tendency.NORMAL
    return Tendency.SLOW


def _speed_category(value: float, q33: float, q67: float) -> Tendency:
    if value <= q33:
        return Tendency.SLOW
    if value <= q67:
        return Tendency.NORMAL
    return Tendency.FAST


def categorize(records: list[EmpiricalRecord]) -> list[EmpiricalRecord]:
    pooled_speed_q = [
        _quantile([record.observed_speed_proxy_mps for record in records], q) for q in (0.33, 0.67)
    ]
    categorized: list[EmpiricalRecord] = []
    for treatment in SocialMode:
        subset = [record for record in records if record.treatment == treatment]
        if len(subset) < 3:
            raise ValueError(f"Treatment {treatment} requires at least three empirical records")
        response_q = [_quantile([r.response_time_s for r in subset], q) for q in (0.33, 0.67)]
        for record in subset:
            categorized.append(
                record.model_copy(
                    update={
                        "response_tendency": _response_category(record.response_time_s, *response_q),
                        "speed_tendency": _speed_category(
                            record.observed_speed_proxy_mps, *pooled_speed_q
                        ),
                    }
                )
            )
    return categorized


class EmpiricalResolver:
    def __init__(self, records: list[EmpiricalRecord], master_seed: int):
        self.records = records
        self.master_seed = master_seed

    def resolve(
        self,
        *,
        persona_uuid: str,
        variant: str,
        treatment: SocialMode,
        response: Tendency,
        speed: Tendency,
    ) -> ResolvedBehavior:
        response_pool = [
            record
            for record in self.records
            if record.treatment == treatment and record.response_tendency == response
        ]
        speed_pool = [record for record in self.records if record.speed_tendency == speed]
        if not response_pool or not speed_pool:
            raise ValueError("Empirical quantile pool is empty")
        response_seed = stable_seed(
            self.master_seed, persona_uuid, variant, RESOLVER_VERSION, "response", treatment, response
        )
        speed_seed = stable_seed(
            self.master_seed, persona_uuid, variant, RESOLVER_VERSION, "speed-pooled", speed
        )
        selected_response = response_pool[random.Random(response_seed).randrange(len(response_pool))]
        selected_speed = speed_pool[random.Random(speed_seed).randrange(len(speed_pool))]
        return ResolvedBehavior(
            response_record_id=selected_response.record_id,
            speed_record_id=selected_speed.record_id,
            response_treatment=treatment,
            response_time_s=selected_response.response_time_s,
            observed_speed_proxy_mps=selected_speed.observed_speed_proxy_mps,
            desired_speed_mps=selected_speed.observed_speed_proxy_mps,
            requested_response_tendency=response,
            requested_speed_tendency=speed,
            resolved_response_tendency=selected_response.response_tendency,
            resolved_speed_tendency=selected_speed.speed_tendency,
            fallback_distance=0.0,
            speed_is_proxy=True,
        )
