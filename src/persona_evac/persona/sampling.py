from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from urllib.request import Request, urlopen

import pyarrow.parquet as pq

from persona_evac.hashing import canonical_json, sha256_value, stable_seed
from persona_evac.schemas import PersonaProfile


def load_personas(path: Path) -> list[PersonaProfile]:
    if path.suffix.lower() == ".parquet":
        rows = pq.read_table(path).to_pylist()
    else:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    profiles = [PersonaProfile.model_validate(row) for row in rows]
    uuids = [profile.uuid for profile in profiles]
    if len(uuids) != len(set(uuids)):
        raise ValueError("Persona source contains duplicate uuid values")
    return profiles


def age_band(age: int) -> str:
    if age < 30:
        return "19-29"
    if age < 45:
        return "30-44"
    if age < 65:
        return "45-64"
    return "65+"


def stratified_sample(profiles: list[PersonaProfile], count: int) -> list[PersonaProfile]:
    """Deterministic largest-remainder sample over age/sex/province strata."""
    if count > len(profiles):
        raise ValueError(f"Need {count} unique personas, only {len(profiles)} are available")
    buckets: dict[tuple[str, str, str], list[PersonaProfile]] = defaultdict(list)
    for profile in profiles:
        buckets[(age_band(profile.age), profile.sex, profile.province)].append(profile)
    for values in buckets.values():
        values.sort(key=lambda item: sha256_value(item.uuid))

    total = len(profiles)
    quotas: dict[tuple[str, str, str], int] = {}
    remainders: list[tuple[float, str, tuple[str, str, str]]] = []
    for key, values in buckets.items():
        exact = count * len(values) / total
        floor = min(len(values), int(exact))
        quotas[key] = floor
        remainders.append((exact - floor, sha256_value(key), key))
    remaining = count - sum(quotas.values())
    for _, _, key in sorted(remainders, reverse=True):
        if remaining == 0:
            break
        if quotas[key] < len(buckets[key]):
            quotas[key] += 1
            remaining -= 1

    selected = [profile for key in sorted(buckets) for profile in buckets[key][: quotas[key]]]
    if len(selected) != count:
        unused = [profile for profile in profiles if profile not in selected]
        unused.sort(key=lambda item: sha256_value(item.uuid))
        selected.extend(unused[: count - len(selected)])
    return sorted(selected, key=lambda item: sha256_value(item.uuid))


def persona_manifest(profiles: list[PersonaProfile], source_ref: str) -> dict[str, object]:
    rows = [profile.model_dump(mode="json") for profile in profiles]
    return {
        "schema_version": "1",
        "source_ref": source_ref,
        "count": len(rows),
        "persona_uuids": [row["uuid"] for row in rows],
        "records_hash": sha256_value(rows),
    }


def fetch_nemotron_snapshot(output: Path, count: int = 10, seed: int = 20260916) -> Path:
    """Sample without replacement from a recorded, paginated source window.

    This is not a representative sample of the entire Korean population/dataset.
    first-rows can truncate to fewer than 100 records, so use the rows endpoint.
    """
    if count < 1:
        raise ValueError("count must be positive")
    rows_url = (
        "https://datasets-server.huggingface.co/rows?"
        "dataset=nvidia%2FNemotron-Personas-Korea&config=default&split=train"
    )
    metadata_url = "https://huggingface.co/api/datasets/nvidia/Nemotron-Personas-Korea"

    def get_json(url: str) -> dict[str, object]:
        request = Request(url, headers={"User-Agent": "persona-evac-mvp/0.1"})
        with urlopen(request, timeout=60) as response:
            return json.load(response)

    metadata = get_json(metadata_url)
    candidates = []
    endpoints = []
    window = max(100, count)
    while len(candidates) < window:
        endpoint = f"{rows_url}&offset={len(candidates)}&length={min(100, window-len(candidates))}"
        payload = get_json(endpoint)
        rows = payload["rows"]
        if not rows:
            break
        endpoints.append(endpoint)
        candidates.extend(PersonaProfile.model_validate(item["row"]) for item in rows)
    if len({p.uuid for p in candidates}) != len(candidates):
        raise ValueError("Dataset source window contains duplicate persona UUIDs")
    if count > len(candidates):
        raise ValueError(f"Dataset returned only {len(candidates)} unique personas")
    revision = str(metadata.get("sha", "unknown"))
    selected = random.Random(stable_seed(seed, revision)).sample(candidates, count)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "\n".join(canonical_json(profile.model_dump(mode="json")) for profile in selected) + "\n",
        encoding="utf-8",
    )
    output.with_suffix(".metadata.json").write_text(
        json.dumps(
            {
                "dataset": "nvidia/Nemotron-Personas-Korea",
                "revision": revision,
                "rows_endpoint": rows_url,
                "pages": endpoints,
                "candidate_count": len(candidates),
                "source_window": [0, len(candidates)],
                "sampling": "seeded random sample without replacement from the recorded source window",
                "count": count,
                "selection_seed": seed,
                "records_hash": sha256_value(
                    [profile.model_dump(mode="json") for profile in selected]
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return output
