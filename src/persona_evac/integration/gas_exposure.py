"""개인 이동 경로에서 적분하는 CO·CO₂·산소 부족 FED. 시야와 독립적이다."""
from __future__ import annotations

import math

FED_THRESHOLD = 1.0
MODEL = {
    'version': 'gas_fed_co_co2_o2_v1',
    'threshold': FED_THRESHOLD,
    'activity': 'light_work',
    'breathing_height_m': 1.5,
    'source': 'https://github.com/firemodels/fds/blob/FDS-6.11.1/Source/func.f90#L2173-L2256',
    'scope': 'CO・CO₂・산소 부족만 반영. 열 상해·HCN·사망은 계산하지 않습니다.',
    'integration': '보행 시간 간격마다 현재 위치·구간 중간 시각의 농도로 적분',
}


def fed_rate_per_s(reading):
    """FDS 6.11.1 FED의 light-work CO, CO₂, O₂ 항; 단위 1/s."""
    try:
        co, co2, o2 = (float(reading[k]) for k in ('co_ppm','co2_percent','o2_percent'))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('FED 계산에 CO·CO₂·산소 농도가 필요합니다. FDS 가스 출력을 확인하세요.') from exc
    if not all(math.isfinite(x) for x in (co,co2,o2)) or not (0 <= co <= 1e6 and 0 <= co2 <= 100 and 0 <= o2 <= 100):
        raise ValueError('FED 가스 농도의 값 또는 단위가 올바르지 않습니다.')
    co_rate = 2.7641667e-5 * co**1.036
    hyperventilation = math.exp(.1903*co2 + 2.0004)/7.1 if co2 > 0 else 1.
    oxygen_rate = math.exp(-8.13 + .54*(20.9-o2)) if o2 < 20 else 0.
    return (co_rate*hyperventilation + oxygen_rate)/60


def accumulate_exposure(runtime, reading, start_s, end_s):
    """정지·대기·반응 전·계단 상태도 누적한다. 대피 완료 후에는 누적하지 않는다.

    같은 구간의 중복 적분을 거부하고 최초 FED>=1 사건을 한 번만 기록한다.
    위치·신체 충돌 처리는 호출한 물리 엔진이 유지한다.
    """
    if runtime['phase'] == 'evacuated':
        return False
    agent = runtime['agent']
    if not (math.isfinite(start_s) and math.isfinite(end_s) and end_s > start_s):
        raise ValueError('가스 노출 적분 시간 구간이 올바르지 않습니다.')
    previous = agent.get('gas_exposure_until_s', 0.)
    if abs(start_s-previous) > 1e-7:
        raise ValueError('가스 노출 적분 구간이 누락되었거나 중복되었습니다.')
    dose = agent.get('fed', 0.) + fed_rate_per_s(reading)*(end_s-start_s)
    agent.update(fed=dose, gas_exposure_until_s=end_s)
    crossed = agent.get('incapacitated_at_s') is None and dose >= FED_THRESHOLD
    if crossed:
        agent['incapacitated_at_s'] = end_s
        agent['incapacitation'] = {
            'time_s': end_s, 'fed': dose, 'threshold': FED_THRESHOLD,
            'cause': 'toxic_gas_exposure', 'previous_phase': runtime['phase'],
            'navigation_blocked': runtime.get('navigation_blocked'),
            'floor': runtime['floor'], 'position': list(runtime['pos']),
        }
        runtime.update(incapacitated=True, passing=False, action='continue', pace='continue')
        if runtime['phase'] != 'on_stairs':
            runtime['phase'] = 'incapacitated'
    history = agent.setdefault('gas_exposure_history', [])
    if not history or end_s-history[-1]['time_s'] >= 1-1e-7 or crossed:
        history.append({'time_s':end_s,'fed':dose,
                        **{k:reading[k] for k in ('co_ppm','co2_percent','o2_percent')}})
    return crossed
