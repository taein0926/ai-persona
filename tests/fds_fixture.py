"""Synthetic reference for axis/time/height sampling, never a live fire default."""
FIRE = {'floor':'B1','rect':[40,13,2,1],'peak_hrr_kw':2000,
        'ramp_full_s':60,'evacuation_start_s':30,'slice_height_m':{'B2':1.5,'B1':5.5}}


def config():
    """축·층·시간·형상 해시 검사만을 위한 독립적인 합성 공간을 반환한다."""
    return {
        'id': 'synthetic-fds-sampling',
        'name': 'FDS 표본 추출 테스트 공간',
        'station_slug': 'synthetic_fds_sampling',
        'floors': [
            {'id': 'B2', 'elevation_m': 0, 'walkable': [[0, 0, 120, 20]]},
            {'id': 'B1', 'elevation_m': 4, 'walkable': [[0, 0, 120, 20]]},
        ],
        'nodes': {
            'lower': {'floor': 'B2', 'x': 60, 'y': 0},
            'upper': {'floor': 'B1', 'x': 60, 'y': 0},
        },
    }
