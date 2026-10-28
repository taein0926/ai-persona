"""Compare local execution settings with unchanged, single-person inputs."""
import argparse
import hashlib
from datetime import datetime,timezone
import json
from pathlib import Path
import time

from persona_evac.integration.private_agents import PrivateController
from persona_evac.llm.ollama import LocalOllama,atomic_json
from persona_evac.llm.private_decisions import PrivateDecisions


class ExperimentalLocal(LocalOllama):
    """Rejected output formats stay in the benchmark, outside the application."""
    def __init__(self,response_format):
        super().__init__();self.response_format=response_format
        if response_format!='full':self.settings['benchmark_response_format']=response_format

    def generate(self,system,body,schema,max_tokens,**kwargs):
        if self.response_format=='full':return super().generate(system,body,schema,max_tokens,**kwargs)
        assert len(body)==1
        aid=str(body[0]['agent_id']);fields={'p':'priority','m':'memory','o':'observations','a':'movement'}
        original=schema['properties'][aid]['properties']
        if self.response_format=='tuple':
            schema={'type':'array','prefixItems':[original[field] for field in fields.values()],'minItems':4,'maxItems':4}
            prompt=system+'\n이번 응답은 한 사람의 선택만 반환한다. 출력은 [priority,memory,observations,movement] 순서의 정확히 4개 항목 배열이다. 각 항목의 의미와 허용 값은 그대로이며 인물 번호나 필드 이름으로 감싸지 않는다.'
        else:
            schema={'type':'object','properties':{key:original[value] for key,value in fields.items()},'required':list(fields),'additionalProperties':False}
            prompt=system+'\n이번 응답은 한 사람의 선택만 반환한다. 출력 키 p=priority, m=memory, o=observations, a=movement이다. 의미와 허용 값은 그대로이며 인물 번호로 감싸지 않는다.'
        payload,metric=super().generate(prompt,body,schema,max_tokens,**kwargs)
        metric.update(transport_raw_response=payload,transport_system_prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest())
        if self.response_format=='tuple':
            if not isinstance(payload,list) or len(payload)!=4:raise ValueError('축약 배열 응답의 항목 수가 다릅니다.')
            return {aid:dict(zip(fields.values(),payload))},metric
        if not isinstance(payload,dict) or set(payload)!=set(fields):raise ValueError('축약 응답의 항목이 다릅니다.')
        return {aid:{value:payload[key] for key,value in fields.items()}},metric


def samples(path,count,offset):
    decisions=json.loads(path.read_text(encoding='utf-8'))['decisions']
    # Different simulation times and people prevent repeated full-prompt hits.
    indices=[round((i+1)*len(decisions)/(count+1))-1 for i in range(count)]
    return [decisions[(index+offset)%len(decisions)]['observation'] for index in indices]


def main():
    parser=argparse.ArgumentParser(description='개인별 입력을 유지하며 로컬 추론 설정을 비교합니다.')
    parser.add_argument('--input',type=Path,default=Path('output/gaehwasan/private_verification/live.json'))
    parser.add_argument('--count',type=int,default=6)
    parser.add_argument('--offset',type=int,default=0)
    parser.add_argument('--batch-sizes',type=int,nargs='+',default=[128,512])
    parser.add_argument('--workers',type=int,nargs='+',default=[1,2])
    parser.add_argument('--rounds',type=int,default=1)
    parser.add_argument('--reset-between-trials',action='store_true')
    parser.add_argument('--server-label',default='설정 미지정')
    parser.add_argument('--compact-response',action='store_true')
    parser.add_argument('--response-format',choices=['full','compact','tuple'])
    args=parser.parse_args()
    if args.count<2 or args.rounds<1:parser.error('표본은 2개 이상, 반복은 1회 이상이어야 합니다.')
    observations=samples(args.input,args.count,args.offset)
    root=Path('output/qa/local_tuning')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    rows=[]
    for round_number in range(args.rounds):
        batch_sizes=args.batch_sizes if round_number%2==0 else list(reversed(args.batch_sizes))
        for batch_size in batch_sizes:
            response_format=args.response_format or ('compact' if args.compact_response else 'full')
            local=ExperimentalLocal(response_format)
            # Load/reconfigure outside timing, using a different person/context.
            warm=samples(args.input,2,args.offset+7)[0]
            PrivateDecisions(root/f'warm_{round_number}_{batch_size}',local,prompt_batch_size=batch_size).decide([warm])
            worker_counts=args.workers if (round_number+args.batch_sizes.index(batch_size))%2==0 else list(reversed(args.workers))
            for workers in worker_counts:
                if args.reset_between_trials:
                    response=local.client.post('/api/generate',json={'model':local.model,'keep_alive':0})
                    response.raise_for_status()
                    PrivateDecisions(root/f'warm_{round_number}_{batch_size}_{workers}',local,prompt_batch_size=batch_size).decide([warm])
                controller=PrivateController.__new__(PrivateController)
                controller.provider=PrivateDecisions(root/f'r{round_number}_b{batch_size}_w{workers}',local,prompt_batch_size=batch_size)
                controller.decision_workers=workers;controller.decision_wall_s=0.0
                # These snapshots are independent. This probes server throughput;
                # a live simulation still batches only within the same tick.
                start=time.perf_counter();outcomes=controller.decide_batch(observations)
                elapsed=time.perf_counter()-start
                metrics=[meta for _,meta in outcomes]
                row={'round':round_number,'prompt_batch_size':batch_size,'workers':workers,
                     'wall_s':elapsed,'people':len(observations),
                     'fallback_count':sum(meta['source']=='fallback' for meta in metrics),
                     'live_requests':sum(meta['source']=='ollama_live' for meta in metrics),
                     'prompt_eval_s':sum(meta.get('prompt_eval_duration',0) for meta in metrics)/1e9,
                     'generation_s':sum(meta.get('eval_duration',0) for meta in metrics)/1e9,
                     'load_s':sum(meta.get('load_duration',0) for meta in metrics)/1e9,
                     'output_tokens':sum(meta.get('eval_count',0) for meta in metrics),
                     'choices':[choices[0]['raw_model_choice'] for choices,_ in outcomes],
                     'metrics':metrics}
                assert row['fallback_count']==0 and row['live_requests']==len(observations)
                assert all(meta['context_people_count']==1 for meta in metrics)
                rows.append(row)
                atomic_json(root/'report.json',{'model':local.model,'digest':local.digest,
                    'source':str(args.input),'sample_offset':args.offset,'reset_between_trials':args.reset_between_trials,
                    'server_label':args.server_label,
                    'compact_response':args.compact_response,
                    'response_format':response_format,
                    'samples':[{'agent_id':o['agent_id'],'time_s':o['time_s'],'events':o['events']} for o in observations],
                    'trials':rows,'note':'고정 입력의 처리량 비교이며 서로 다른 모의 시점을 실제 실행에서 묶지 않습니다.'})
                print(json.dumps({k:v for k,v in row.items() if k not in {'metrics','choices'}},ensure_ascii=False),flush=True)
    print(json.dumps({'report':str(root/'report.json')},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
