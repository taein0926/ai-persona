"""Measure isolated local requests without replacing verified simulation results."""
import argparse
from collections import defaultdict
from datetime import datetime,timezone
import json
from pathlib import Path
from threading import Lock
import time

from persona_evac.integration.private_agents import PrivateController
from persona_evac.llm.ollama import LocalOllama,atomic_json
from persona_evac.llm.private_decisions import PrivateDecisions


class MeasuredLocal(LocalOllama):
    def __init__(self):
        super().__init__()
        self.lock=Lock();self.active=0;self.peak=0

    def generate(self,*args,**kwargs):
        with self.lock:
            self.active+=1;self.peak=max(self.peak,self.active)
        try:return super().generate(*args,**kwargs)
        finally:
            with self.lock:self.active-=1


def main():
    parser=argparse.ArgumentParser(description='로컬 Qwen의 개인별 병렬 요청을 검증합니다.')
    parser.add_argument('--input',type=Path,default=Path('output/gaehwasan/private_verification/live.json'))
    parser.add_argument('--count',type=int,default=4)
    parser.add_argument('--rounds',type=int,default=2)
    args=parser.parse_args()
    if args.count<2 or args.rounds<1:parser.error('인원은 2명 이상, 반복은 1회 이상이어야 합니다.')
    groups=defaultdict(list)
    for d in json.loads(args.input.read_text(encoding='utf-8'))['decisions']:
        groups[d['time_s']].append(d['observation'])
    observations=next(group[:args.count] for group in groups.values() if len(group)>=args.count)
    run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    root=Path('output/qa/local_parallel')/run_id
    local=MeasuredLocal()
    # Warm up outside the measured rounds; each round has its own response cache.
    PrivateDecisions(root/'warmup',local).decide(observations[:1])
    rounds=[]
    for round_number in range(args.rounds):
        for workers in ((1,2) if round_number%2==0 else (2,1)):
            controller=PrivateController.__new__(PrivateController)
            controller.provider=PrivateDecisions(root/f'round_{round_number}_workers_{workers}',local)
            controller.decision_workers=workers;controller.decision_wall_s=0.0;local.peak=0
            started=time.perf_counter();results=controller.decide_batch(observations)
            wall_s=time.perf_counter()-started
            assert [choices[0]['agent_id'] for choices,_ in results]==[o['agent_id'] for o in observations]
            assert all(meta['source']=='ollama_live' and meta['context_people_count']==1 for _,meta in results)
            attempts_before=len(local.attempts)
            cached=controller.decide_batch(observations)
            assert len(local.attempts)==attempts_before
            assert all(meta['source']=='ollama_cache' for _,meta in cached)
            assert [choices for choices,_ in results]==[choices for choices,_ in cached]
            row={'round':round_number,'workers':workers,'wall_s':wall_s,'peak_client_requests':local.peak,
                 'people':len(observations),'context_people_count':1,'live_requests':len(results),
                 'cache_replay_requests':len(cached),'fallback_count':0,
                 'choices':[choices[0]['raw_model_choice'] for choices,_ in results]}
            rounds.append(row);print(json.dumps(row,ensure_ascii=False),flush=True)
    means={workers:sum(r['wall_s'] for r in rounds if r['workers']==workers)/args.rounds for workers in (1,2)}
    report={'model':local.model,'model_digest':local.digest,'settings':local.settings,
            'source':str(args.input),'source_time_s':observations[0]['time_s'],
            'subject_agent_ids':[o['agent_id'] for o in observations],'rounds':rounds,
            'sequential_mean_s':means[1],'parallel_mean_s':means[2],'speedup':means[1]/means[2],
            'note':'소규모 로컬 요청 측정이며 전체 시뮬레이션의 단축률을 보장하지 않습니다.'}
    atomic_json(root/'report.json',report)
    print(json.dumps({'report':str(root/'report.json'),'speedup':report['speedup']},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
