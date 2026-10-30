"""Selection arithmetic over private, explicitly sourced alternatives only."""
from persona_evac.agent_policy import POLICY, PROFILE_CLASSIFICATION


def observed_flow_strength(observation,option):
    flow=option.get('observed_flow') or {}
    fresh=0<=observation.get('time_s',0)-flow.get('observed_at_s',-1e9)<=POLICY['sensor']['current_observation_ttl_s']
    return max(0,min(1,flow.get('strength',0))) if fresh and flow.get('source')=='own_observed_movement' else 0


def resolve_private(observation,choice):
    traits=observation['profile']['traits'];priority=choice['priority'];rows=[]
    params=POLICY['trait_parameters']
    crowd=params['crowd_seconds'][traits['crowd_avoidance']['level']]
    patience=params['patience_seconds'][traits['waiting_patience']['level']]
    risk=params['risk_multiplier'][traits['risk_avoidance']['level']]
    explore=params['exploration_bonus'][traits['exploration']['level']]
    following_level=traits.get('social_following',{}).get('level','low')
    following=PROFILE_CLASSIFICATION['full_persona']['following_bonus_seconds'][following_level]
    for option in observation['options']:
        # Missing observations remain unknown; they are never filled from engine state.
        people=option['visible_people'];h=option['hazard'] or {}
        uncertainty=0 if option['state_is_current'] else max(1,12-explore)
        crowd_cost=0 if people is None else crowd*min(1,people/12)**2
        hazard_cost=risk*({'clear':0,'light':3,'dense':15,'opaque':45}.get(h.get('smoke'),0)+
                          {'normal':0,'warm':8,'hot':45}.get(h.get('heat'),0))
        past=(option.get('past_state') or {}).get('hazard') or {}
        confidence=max(.25,1-(option.get('age_s') or 0)/60)
        historical_hazard_cost=0 if option['state_is_current'] else risk*confidence*(
            {'clear':0,'light':3,'dense':15,'opaque':45}.get(past.get('smoke'),0)+{'normal':0,'warm':8,'hot':45}.get(past.get('heat'),0))
        own_wait=observation['queue_wait_s'] if option['id']==observation['current_target'] else 0
        impatience=max(0,own_wait-patience)*2
        remembered=sum(max(m['waited_s'],m.get('stalled_s',0)) for m in observation['memory'] if m['id'] in choice['memory']
                       and m['stair_id']==option['id'] and m['result']=='still_blocked')
        frontier=option['kind']=='frontier'
        novelty=explore*option.get('novelty',0) if frontier else 0
        revisit=option.get('visited',0)*12 if frontier else 0
        bearing_bonus=max(-18,min(18,option.get('remembered_bearing_progress_m',0))) if frontier else 0
        following_bonus=following*observed_flow_strength(observation,option)
        # Ascending stairs are an observed evacuation affordance, not a known exit.
        upward_bonus=18 if not frontier else 0
        if priority=='crowding':crowd_cost*=1.5
        if priority=='risk':hazard_cost*=1.5;historical_hazard_cost*=1.5
        if priority=='explore':novelty*=1.5
        if priority=='patience':impatience*=1.5
        if priority=='experience':remembered*=1.5
        if priority=='follow':following_bonus*=1.5
        walking_cost=option['walking_s']*(1.5 if priority=='time' else 1)
        score=walking_cost+crowd_cost+hazard_cost+historical_hazard_cost+uncertainty+impatience+remembered+revisit-novelty-upward_bonus-bearing_bonus-following_bonus
        rows.append({'target_id':option['id'],'score':round(score,4),'walking_s':option['walking_s'],
                     'crowd_cost':round(crowd_cost,3),'hazard_cost':round(hazard_cost,3),'uncertainty_cost':uncertainty,
                     'historical_hazard_cost':round(historical_hazard_cost,3),
                     'own_wait_cost':impatience,'memory_cost':remembered,'exploration_bonus':novelty,
                     'revisit_cost':revisit,'upward_affordance_bonus':upward_bonus,
                     'walking_cost':walking_cost,'remembered_bearing_bonus':bearing_bonus,
                     'following_bonus':round(following_bonus,3),
                     'knowledge_observation_ids':option['observation_ids']})
    if not rows:return None,rows
    current=observation['current_target']
    if choice['movement'] in {'wait','overtake'} and any(r['target_id']==current for r in rows):target=current
    else:target=min(rows,key=lambda r:(r['score'],r['target_id']!=current,r['target_id']))['target_id']
    return target,rows
