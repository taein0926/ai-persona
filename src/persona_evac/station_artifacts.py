"""Signatures describe code that actually computed an artifact, never a retagging."""

import hashlib
from persona_evac.agent_policy import ROOT, VERSIONS, POLICY_HASH
from persona_evac.integration.personal_space import PRIOR_HASH


def station_signature():
    names = [
        "agent_policy.py",
        "hashing.py",
        "station_artifacts.py",
        "integration/station.py",
        "integration/station_dynamics.py",
        "integration/station_connectors.py",
        "integration/station_geometry.py",
        "integration/station_layout.py",
        "integration/stair_dynamics.py",
        "integration/agent_memory.py",
        "integration/private_agents.py",
        "integration/personal_space.py",
        "integration/private_strategy.py",
        "integration/fds_environment.py",
        "integration/fds_station_geometry.py",
        "persona/experimental_profile.py",
        "persona/full_profile.py",
        "persona/station_population.py",
        "llm/private_decisions.py",
        "llm/ollama.py",
        "llm/common.py",
        "llm/baseline_profile.py",
    ]
    paths = [ROOT / "src/persona_evac" / name for name in names]
    paths += [
        ROOT / "configs/scenarios/gaehwasan.json",
        *sorted((ROOT / "configs/agents").glob("*.json")),
    ]
    digest = hashlib.sha256()
    for p in paths:
        digest.update(str(p.relative_to(ROOT)).replace("\\", "/").encode())
        digest.update(b"\0")
        digest.update(p.read_bytes())
    return digest.hexdigest()


def policy_matches(result):
    policy = result.get("agent_policy", {})
    method = policy.get('classification_method', 'keyword_candidates')
    versions = VERSIONS
    if method == 'full_persona':
        from persona_evac.persona.full_profile import CONFIG, POLICY_HASH_FULL
        versions = {**VERSIONS, 'profile': CONFIG['version']}
        if policy.get('profile_policy_hash') != POLICY_HASH_FULL:
            return False
    elif method != 'keyword_candidates':
        return False
    return (
        policy.get("versions") == versions
        and policy.get("policy_hash") == POLICY_HASH
        and policy.get("prior_hash") == PRIOR_HASH
    )
