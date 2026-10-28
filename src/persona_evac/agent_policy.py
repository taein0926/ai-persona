"""Versioned experimental assumptions; no physical simulator replacements."""
import hashlib
import json
from pathlib import Path

PACKAGE_ROOT=Path(__file__).resolve().parents[2]
ROOT=PACKAGE_ROOT if (PACKAGE_ROOT/'configs/agents/private_agent_policy.json').exists() else Path.cwd()
POLICY=json.loads((ROOT/'configs/agents/private_agent_policy.json').read_text(encoding='utf-8'))
VERSIONS=POLICY['versions']
# Classification extensions have their own hash. Preserve the original policy hash,
# UUID draws and existing profile/decision caches byte for byte.
POLICY_HASH=hashlib.sha256(json.dumps({k:v for k,v in POLICY.items() if k!='profile_classification'},sort_keys=True,ensure_ascii=False).encode()).hexdigest()
PROFILE_CLASSIFICATION=POLICY['profile_classification']
