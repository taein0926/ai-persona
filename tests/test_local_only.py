"""클라우드 SDK 없이 로컬 추론·기존 재생 자료를 사용할 수 있는지 검사한다."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from persona_evac.integration.runner import run_experiment
from persona_evac.persona.station_population import prepare_population


def test_application_imports_without_cloud_sdk():
    code = """
import importlib.abc
import sys
class BlockCloudSDK(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'openai':
            raise ImportError('클라우드 SDK 의존성이 남아 있습니다.')
sys.meta_path.insert(0, BlockCloudSDK())
import persona_evac.visualization.app
import persona_evac.cli
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


@pytest.mark.parametrize('provider', ['openai', 'groq', 'unsupported'])
def test_experiment_rejects_removed_providers(tmp_path, monkeypatch, provider):
    config = json.loads(Path('configs/experiments/mvp.json').read_text(encoding='utf-8'))
    config['output_dir'] = str(tmp_path)
    path = tmp_path/'config.json'
    path.write_text(json.dumps(config), encoding='utf-8')
    monkeypatch.setenv('LLM_PROVIDER', provider)
    monkeypatch.setenv('PERSONA_EVAC_OUTPUT', str(tmp_path))
    with pytest.raises(ValueError, match='로컬 Ollama'):
        run_experiment(path)


def test_population_rejects_cloud_provider_before_inference(tmp_path, monkeypatch):
    monkeypatch.setenv('LLM_PROVIDER', 'groq')
    with pytest.raises(ValueError, match='로컬 Ollama'):
        prepare_population(
            Path('data/raw/nemotron_personas_100.jsonl'),
            Path('data/processed/julich_features.csv'), tmp_path/'source.json',
        )
    assert not (tmp_path/'source.json').exists()


def test_legacy_result_provenance_remains_readable():
    from persona_evac.schemas import ExperimentResults
    payload = json.loads(Path('output/t-junction-mvp-openai/seed-1/pyfds_results.json').read_text(encoding='utf-8'))
    ExperimentResults.model_validate(payload)
