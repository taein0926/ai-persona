from copy import deepcopy
import json
import shutil
import pytest
from persona_evac.agent_policy import VERSIONS,POLICY_HASH
from persona_evac.integration.personal_space import PRIOR_HASH
from persona_evac import station_artifacts as artifacts


@pytest.mark.parametrize('changed',['profile','knowledge','observation','decision'])
def test_old_or_changed_policy_cannot_be_accepted_as_current(changed):
    current={'agent_policy':{'versions':deepcopy(VERSIONS),'policy_hash':POLICY_HASH,'prior_hash':PRIOR_HASH}}
    assert artifacts.policy_matches(current)
    old=deepcopy(current);old['agent_policy']['versions'][changed]='old_policy'
    assert not artifacts.policy_matches(old)
    assert not artifacts.policy_matches({'behavior':{'policy':{'version':'persona_memory_events_v1'}}})
    old=deepcopy(current);old['agent_policy']['prior_hash']='different_prior'
    assert not artifacts.policy_matches(old)


def test_knowledge_implementation_and_trait_distribution_invalidate_signature(tmp_path,monkeypatch):
    shutil.copytree(artifacts.ROOT/'src',tmp_path/'src',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(artifacts.ROOT/'configs',tmp_path/'configs')
    monkeypatch.setattr(artifacts,'ROOT',tmp_path)
    first=artifacts.station_signature()
    space=tmp_path/'src/persona_evac/integration/personal_space.py'
    space.write_bytes(space.read_bytes()+b'\n# signature regression fixture\n')
    second=artifacts.station_signature();assert second!=first
    policy=tmp_path/'configs/agents/private_agent_policy.json';value=json.loads(policy.read_text(encoding='utf-8'))
    value['fallback_distribution']={'low':.1,'normal':.65,'high':.25}
    policy.write_text(json.dumps(value,ensure_ascii=False),encoding='utf-8')
    assert artifacts.station_signature()!=second
