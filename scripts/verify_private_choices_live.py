"""Actual Qwen calls on disclosed synthetic controls, separate from the 100 originals."""
from copy import deepcopy
import json
from pathlib import Path
from persona_evac.agent_policy import VERSIONS
from persona_evac.persona.experimental_profile import build_profile,TRAITS
from persona_evac.integration.personal_space import residence_basis
from persona_evac.llm.private_decisions import PrivateDecisions,project_private
from persona_evac.llm.ollama import atomic_json

person={'uuid':'synthetic-controlled-agent','province':'부산','district':'부산-강서구'}
base=build_profile(person)
for value in base['traits'].values():
    value.update(level='normal',origin='controlled_test_assignment',evidence_ids=[],assumption='원본 100명과 분리한 통제 시험 조건')

def option(id,walk=5,people=0,smoke='clear',kind='up_stair',novelty=0):
    return {'id':id,'kind':kind,'walking_s':walk,'distance_m':walk,'position':[walk,0],
        'visible_people':people,'hazard':{'smoke':smoke,'heat':'unknown'},'past_state':None,
        'state_is_current':True,'age_s':0,'known_to_floor':None,'source':'controlled_observation',
        'observation_ids':['O1'],'novelty':novelty,'path':[]}

def scene():
    return {'agent_id':1,'time_s':60,'events':['new_route'],'floor':'B2','phase':'walking',
        'profile':deepcopy(base),'familiarity':residence_basis(person),'current_target':'near','memory':[],
        'observations':[{'id':'O1','time_s':60,'source':'vision','kind':'controlled_visible_routes'}],
        'options':[option('near',3),option('far',12)],'local_hazard':{'smoke':'clear','heat':'normal'},
        'queue_wait_s':0,'allowed_movements':['continue','hurry'],
        'private_knowledge':{'local_ids':{'near':'L1','far':'L2'},'known_landmarks':[]}}

cases=[]
for trait in TRAITS:
    o=scene()
    if trait=='crowd_avoidance':o['options'][0]['visible_people']=12
    if trait=='waiting_patience':o['queue_wait_s']=14;o['events']=['queue_delayed']
    if trait=='risk_avoidance':o['options'][0]['hazard']['smoke']='dense';o['options'][1]['walking_s']=30;o['events']=['risk_changed']
    if trait=='exploration':o['options']=[option('near',3),option('far',14,kind='frontier',novelty=4)]
    for level in ('low','high'):
        case=deepcopy(o);case['profile']['traits'][trait]['level']=level
        cases.append((trait+'_'+level,case))
one=scene();cases.append(('memory_without',one));two=deepcopy(one)
two['memory']=[{'id':'M1','time_s':30,'action':'navigate','stair_id':'near','floor':'B2',
    'elapsed_s':30,'walked_m':.2,'waited_s':20,'stalled_s':20,'result':'still_blocked','selection_source':'controlled_experience'}]
cases.append(('memory_after_block',two))
one=scene();one['options']=[option('near',12,kind='frontier',novelty=2),option('far',12,kind='frontier',novelty=2)]
cases.append(('knowledge_first_visit',one));two=deepcopy(one)
two['familiarity']=residence_basis({'uuid':person['uuid'],'province':'서울','district':'서울-강서구'})
two['private_knowledge']['known_landmarks']=[{'id':'past_link','floor':'B2','to_floor':'B1'}]
two['private_knowledge']['local_ids']['past_link']='L3'
two['options'][0]['remembered_bearing_progress_m']=-10;two['options'][1]['remembered_bearing_progress_m']=10
cases.append(('knowledge_historical_bearing',two))
provider=PrivateDecisions('output/gaehwasan/private_controlled_final_cache');records=[]
target=Path('output/gaehwasan/private_verification/controlled_choices.json')
for name,o in cases:
    choices,meta=provider.decide([o]);records.append({'name':name,'input':project_private(o),'observation':o,'decision':choices[0],'inference':meta})
    atomic_json(target,{'status':'진행 중','input_origin':'명시한 합성 통제 시험. 원본 100명·실제 관측 자료와 별도. 행동 차이를 강제하지 않음.','cases':records})
    print(json.dumps({'case':name,'raw_choice':choices[0]['raw_model_choice'],'final_target':choices[0]['final_target'],'source':meta['source']},ensure_ascii=False),flush=True)
pairs=[]
for i in range(0,len(records),2):
    a,b=records[i:i+2]
    pairs.append({'pair':a['name'].rsplit('_',1)[0],
        'model_choice_changed':a['decision']['raw_model_choice']!=b['decision']['raw_model_choice'],
        'executed_goal_changed':a['decision']['final_target']!=b['decision']['final_target'],
        'before':a['decision']['final_target'],'after':b['decision']['final_target']})
atomic_json(target,{'status':'완료','input_origin':'명시한 합성 통제 시험. 원본 100명·실제 관측 자료와 별도. 행동 차이를 강제하지 않음.',
                   'policy_versions':VERSIONS,'pairs':pairs,'cases':records,'attempts':provider.local.attempts,
                   'inference_settings':provider.local.settings})
print(json.dumps({'pairs':pairs},ensure_ascii=False))
