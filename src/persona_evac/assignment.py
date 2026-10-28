from __future__ import annotations

import json
import random
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from persona_evac.hashing import canonical_json, sha256_value, stable_seed
from persona_evac.schemas import AssignmentRow, PersonaProfile, RuntimeAgentRow


ASSIGNMENT_VERSION = "1"


def build_assignment(
    *,
    experiment_id: str,
    master_seed: int,
    replicate_seed: int,
    profiles: list[PersonaProfile],
    persona_manifest_hash: str,
    agent_count: int,
) -> tuple[list[AssignmentRow], str]:
    if agent_count > len(profiles):
        raise ValueError(f"Need {agent_count} unique personas, only {len(profiles)} are available")
    uuids = [profile.uuid for profile in profiles]
    if len(uuids) != len(set(uuids)):
        raise ValueError("Cannot assign duplicate persona UUIDs")
    assignment_seed = stable_seed(
        master_seed, replicate_seed, persona_manifest_hash, ASSIGNMENT_VERSION
    ) & ((1 << 63) - 1)
    shuffled = uuids[:agent_count]
    random.Random(assignment_seed).shuffle(shuffled)
    rows = [
        AssignmentRow(
            experiment_id=experiment_id,
            replicate_seed=replicate_seed,
            assignment_seed=assignment_seed,
            agent_slot=slot,
            persona_uuid=persona_uuid,
            persona_manifest_hash=persona_manifest_hash,
        )
        for slot, persona_uuid in enumerate(shuffled)
    ]
    validate_assignment(rows, agent_count)
    return rows, assignment_hash(rows)


def assignment_hash(rows: list[AssignmentRow]) -> str:
    return sha256_value([row.model_dump(mode="json") for row in rows])


def validate_assignment(rows: list[AssignmentRow], expected_count: int) -> None:
    if len(rows) != expected_count:
        raise ValueError(f"Assignment has {len(rows)} rows, expected {expected_count}")
    slots = [row.agent_slot for row in rows]
    uuids = [row.persona_uuid for row in rows]
    if slots != list(range(expected_count)) or len(set(slots)) != expected_count:
        raise ValueError("Assignment agent_slot values are not a complete unique sequence")
    if len(set(uuids)) != expected_count:
        raise ValueError("Assignment must be a one-to-one Persona mapping")
    invariant = {
        (row.experiment_id, row.replicate_seed, row.assignment_seed, row.persona_manifest_hash)
        for row in rows
    }
    if len(invariant) != 1:
        raise ValueError("Assignment rows disagree on experiment provenance")


def bind_runtime_agents(
    rows: list[AssignmentRow], runtime_ids: list[int], variant: str, manifest_hash: str
) -> list[RuntimeAgentRow]:
    validate_assignment(rows, len(rows))
    if len(runtime_ids) != len(rows) or len(set(runtime_ids)) != len(runtime_ids):
        raise ValueError("Runtime Agent IDs must be unique and match the assignment count")
    if manifest_hash != assignment_hash(rows):
        raise ValueError("Assignment manifest hash mismatch")
    return [
        RuntimeAgentRow(
            agent_slot=row.agent_slot,
            jupedsim_agent_id=runtime_ids[row.agent_slot],
            persona_uuid=row.persona_uuid,
            variant=variant,
            replicate_seed=row.replicate_seed,
            assignment_manifest_hash=manifest_hash,
        )
        for row in rows
    ]


def write_rows(path: Path, rows: list[AssignmentRow] | list[RuntimeAgentRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [row.model_dump(mode="json") for row in rows]
    if path.suffix == ".parquet":
        pq.write_table(pa.Table.from_pylist(payload), path, compression="NONE", version="2.6")
    else:
        path.write_text(canonical_json(payload) + "\n", encoding="utf-8")


def write_assignment_bundle(directory: Path, rows: list[AssignmentRow]) -> str:
    digest = assignment_hash(rows)
    write_rows(directory / "assignment_manifest.parquet", rows)
    (directory / "assignment_manifest.json").write_text(
        json.dumps(
            {"assignment_manifest_hash": digest, "rows": [r.model_dump(mode="json") for r in rows]},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return digest
