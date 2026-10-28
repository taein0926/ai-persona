from __future__ import annotations

from pathlib import Path

from persona_evac.assignment import bind_runtime_agents, build_assignment, write_rows
from persona_evac.schemas import PersonaProfile


def profiles(count: int = 10) -> list[PersonaProfile]:
    return [
        PersonaProfile(uuid=f"persona-{index}", persona="fixture", age=20 + index)
        for index in range(count)
    ]


def build(seed: int = 1):
    return build_assignment(
        experiment_id="test",
        master_seed=99,
        replicate_seed=seed,
        profiles=profiles(),
        persona_manifest_hash="manifest",
        agent_count=10,
    )


def test_assignment_is_reproducible_unique_and_seeded() -> None:
    first, first_hash = build(1)
    second, second_hash = build(1)
    other, _ = build(2)
    assert first == second
    assert first_hash == second_hash
    assert [row.persona_uuid for row in first] != [row.persona_uuid for row in other]
    assert len({row.persona_uuid for row in first}) == 10
    assert [row.agent_slot for row in first] == list(range(10))
    assert 0 <= first[0].assignment_seed <= (1 << 63) - 1


def test_assignment_parquet_is_byte_identical(tmp_path: Path) -> None:
    rows, _ = build()
    first, second = tmp_path / "first.parquet", tmp_path / "second.parquet"
    write_rows(first, rows)
    write_rows(second, rows)
    assert first.read_bytes() == second.read_bytes()


def test_runtime_binding_is_fail_closed() -> None:
    rows, digest = build()
    a = bind_runtime_agents(rows, list(range(100, 110)), "A", digest)
    c = bind_runtime_agents(rows, list(range(300, 310)), "C", digest)
    assert [(row.agent_slot, row.persona_uuid) for row in a] == [
        (row.agent_slot, row.persona_uuid) for row in c
    ]
    try:
        bind_runtime_agents(rows, list(range(9)), "A", digest)
    except ValueError as exc:
        assert "match" in str(exc)
    else:
        raise AssertionError("Agent count mismatch must fail")
    try:
        bind_runtime_agents(rows, list(range(10)), "A", "wrong")
    except ValueError as exc:
        assert "hash" in str(exc)
    else:
        raise AssertionError("Manifest hash mismatch must fail")


def test_persona_shortage_fails() -> None:
    try:
        build_assignment(
            experiment_id="short",
            master_seed=1,
            replicate_seed=1,
            profiles=profiles(2),
            persona_manifest_hash="manifest",
            agent_count=3,
        )
    except ValueError as exc:
        assert "only 2" in str(exc)
    else:
        raise AssertionError("Persona shortage must fail")
