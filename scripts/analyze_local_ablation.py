"""Attribute execution effects only when exactly one setting changes."""
import argparse
import json
from pathlib import Path
from statistics import mean

from persona_evac.llm.ollama import atomic_json


def trial_settings(row):
    profile=row['metrics'][0]['runtime_profile']
    return {**{'runtime.'+key:value for key,value in profile.items()},
            'prompt_batch_size':row['prompt_batch_size'],'workers':row['workers']}


def main():
    parser=argparse.ArgumentParser(description='추론 설정을 하나씩만 바꾼 실험의 효과를 비교합니다.')
    parser.add_argument('reference',type=Path)
    parser.add_argument('candidates',type=Path,nargs='+')
    parser.add_argument('--output',type=Path,default=Path('output/qa/local_tuning/ablation_summary.json'))
    args=parser.parse_args()
    reference=json.loads(args.reference.read_text(encoding='utf-8'))
    reference_rows=reference['trials']
    reference_settings=trial_settings(reference_rows[0])
    reference_mean=mean(row['wall_s'] for row in reference_rows)
    reference_hashes=[m['input_sha256'] for m in reference_rows[0]['metrics']]
    effects=[]
    for path in args.candidates:
        candidate=json.loads(path.read_text(encoding='utf-8'))
        assert candidate['digest']==reference['digest'] and candidate['samples']==reference['samples']
        settings=trial_settings(candidate['trials'][0])
        changes={key:{'current':reference_settings.get(key),'comparison':settings.get(key)}
                 for key in set(reference_settings)|set(settings) if reference_settings.get(key)!=settings.get(key)}
        assert len(changes)==1,changes
        for row in candidate['trials']:
            assert trial_settings(row)==settings
            assert row['fallback_count']==0 and row['live_requests']==len(reference_hashes)
            assert [m['input_sha256'] for m in row['metrics']]==reference_hashes
            assert all(m['context_people_count']==1 for m in row['metrics'])
        elapsed=mean(row['wall_s'] for row in candidate['trials'])
        effects.append({'report':str(path),'only_changed_setting':changes,
                        'comparison_mean_s':elapsed,'current_mean_s':reference_mean,
                        'extra_seconds_when_reverted':elapsed-reference_mean,
                        'current_time_reduction_percent':100*(1-reference_mean/elapsed),
                        'trial_wall_s':[row['wall_s'] for row in candidate['trials']],
                        'output_tokens':[row['output_tokens'] for row in candidate['trials']]})
    report={'reference':str(args.reference),'reference_settings':reference_settings,
            'reference_mean_s':reference_mean,'reference_trial_wall_s':[r['wall_s'] for r in reference_rows],
            'unique_situations':len(reference['samples']),'samples':reference['samples'],
            'identical_individual_inputs':True,'context_people_count':1,'fallback_count':0,'effects':effects,
            'limitations':['현재 설정에서 한 항목씩 되돌린 효과이며 항목별 단축률을 더할 수 없습니다.',
                           '고정 입력의 추론 측정이며 전체 시뮬레이션의 단축률이나 모델 판단 동일성을 보장하지 않습니다.']}
    atomic_json(args.output,report)
    print(json.dumps({'report':str(args.output),'reference_mean_s':reference_mean,'effects':effects},ensure_ascii=False))


if __name__=='__main__':main()
