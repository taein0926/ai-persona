"""Compare measured settings only when the individual inputs match exactly."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path

from persona_evac.llm.ollama import atomic_json


def main():
    parser=argparse.ArgumentParser(description='같은 개인별 입력을 사용한 추론 측정값을 비교합니다.')
    parser.add_argument('baseline',type=Path)
    parser.add_argument('candidate',type=Path)
    parser.add_argument('--batch-size',type=int,default=128)
    parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--candidate-batch-size',type=int)
    parser.add_argument('--candidate-workers',type=int)
    args=parser.parse_args()
    baseline=json.loads(args.baseline.read_text(encoding='utf-8'))
    candidate=json.loads(args.candidate.read_text(encoding='utf-8'))
    assert baseline['digest']==candidate['digest'] and baseline['samples']==candidate['samples']
    candidate_batch_size=args.candidate_batch_size or args.batch_size
    candidate_workers=args.candidate_workers or args.workers
    def select(report,batch_size,workers):
        return [r for r in report['trials'] if r['workers']==workers and r['prompt_batch_size']==batch_size]
    old,new=select(baseline,args.batch_size,args.workers),select(candidate,candidate_batch_size,candidate_workers)
    assert old and new
    for before,after in zip(old,new):
        assert [m['input_sha256'] for m in before['metrics']]==[m['input_sha256'] for m in after['metrics']]
        assert before['fallback_count']==after['fallback_count']==0
        assert all(m['context_people_count']==1 for m in before['metrics']+after['metrics'])
        assert all(m['source']=='ollama_live' for m in before['metrics']+after['metrics'])
    means={label:sum(r['wall_s'] for r in rows)/len(rows) for label,rows in [('baseline',old),('candidate',new)]}
    comparisons=[{'agent_id':sample['agent_id'],'time_s':sample['time_s'],
                  'raw_choice_equal':before==after,'strategy_equal':before['priority']==after['priority'],
                  'movement_equal':before['movement']==after['movement']}
                 for sample,before,after in zip(baseline['samples'],old[0]['choices'],new[0]['choices'])]
    report={'baseline':str(args.baseline),'candidate':str(args.candidate),'baseline_batch_size':args.batch_size,
            'candidate_batch_size':candidate_batch_size,'baseline_workers':args.workers,'candidate_workers':candidate_workers,
            'baseline_runtime_profile':old[0]['metrics'][0].get('runtime_profile'),
            'candidate_runtime_profile':new[0]['metrics'][0].get('runtime_profile'),
            'model_digest':baseline['digest'],
            'baseline_trial_wall_s':[r['wall_s'] for r in old],
            'candidate_trial_wall_s':[r['wall_s'] for r in new],
            'people':len(baseline['samples']),'identical_individual_inputs':True,
            'context_people_count':1,'fallback_count':0,'baseline_wall_s':means['baseline'],
            'candidate_wall_s':means['candidate'],'speedup':means['baseline']/means['candidate'],
            'wall_time_reduction_percent':100*(1-means['candidate']/means['baseline']),
            'baseline_output_tokens':old[0]['output_tokens'],'candidate_output_tokens':new[0]['output_tokens'],
            'raw_choice_matches':sum(c['raw_choice_equal'] for c in comparisons),
            'strategy_matches':sum(c['strategy_equal'] for c in comparisons),
            'movement_matches':sum(c['movement_equal'] for c in comparisons),'choice_comparisons':comparisons,
            'limitations':['고정 입력 표본의 추론 측정이며 전체 100명 시뮬레이션을 재실행한 결과가 아닙니다.',
                           '실행기의 수치 연산이나 출력 형식 변경은 모델 선택에 영향을 줄 수 있으며 입력 보존이 판단 동일성을 보장하지 않습니다.']}
    path=Path('output/qa/local_tuning')/('comparison_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json')
    atomic_json(path,report)
    print(json.dumps({k:v for k,v in report.items() if k not in {'choice_comparisons','limitations'}},ensure_ascii=False))
    print(json.dumps({'report':str(path)},ensure_ascii=False))


if __name__=='__main__':main()
