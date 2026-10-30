"""Validate and visualize the completed fire-only run before activation."""
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from shapely import contains_xy
from shapely.geometry import Polygon
from shapely.ops import unary_union

from persona_evac.integration.fds_environment import FireField,sha
from persona_evac.integration.station_geometry import floor_geometry

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'output/fds/gaehwasan_20261029_layout12'


def validate():
    snapshot=RUN/'scenario.json';config=json.loads(snapshot.read_text(encoding='utf8'))
    manifest=json.loads((RUN/'run_manifest.json').read_text(encoding='utf8'))
    assert sha(snapshot)==manifest['scenario_sha256']==sha(ROOT/'configs/scenarios/gaehwasan.json')
    assert sha(RUN/'gaehwasan.fds')==manifest['deck_sha256']==sha(ROOT/'configs/fds/gaehwasan.fds')
    field=FireField(config,RUN/'derived',RUN/'gaehwasan.fds')
    text=(RUN/'gaehwasan.out').read_text(errors='replace')
    assert text.count('FDS completed successfully')==1 and '\x00' not in text
    assert field.times[-1]>=600-.01 and np.all(np.diff(field.times)>0)
    assert all(np.isfinite(a).all() for a in field.data.values())
    assert len(field.data['volume_z'])==12
    xx,yy=np.meshgrid(field.data['x'],field.data['y'],indexing='ij')
    stats={};curves={};masks={}
    for floor in config['floors']:
        fid=floor['id'];mask=contains_xy(floor_geometry(floor),xx,yy);masks[fid]=mask
        k=field.data[fid+'_k'][:,mask];temperature=field.data[fid+'_temperature'][:,mask]
        assert k.min()>=0 and k.max()>0
        curve=k.max(axis=1);temp_curve=temperature.max(axis=1);curves[fid]=(curve,temp_curve)
        reached=np.flatnonzero(curve>.01)
        stats[fid]={'max_extinction_per_m':float(curve.max()),'max_temperature_c':float(temp_curve.max()),
            'first_above_k_0_01_s':float(field.times[reached[0]]) if len(reached) else None,
            'max_k_at_600_s':float(curve[-1]),'max_temperature_at_600_c':float(temp_curve[-1])}
    before=manifest['previous_passenger_results_sha256']
    assert sha(ROOT/'output/gaehwasan/results.json')==before
    elapsed=re.search(r'Total Elapsed Wall Clock Time \(s\):\s*([\d.]+)',text)
    hrr=np.loadtxt(RUN/'gaehwasan_hrr.csv',delimiter=',',skiprows=2)
    assert np.isfinite(hrr).all() and hrr[-1,0]>=599 and hrr[:,1].max()>0
    report={'status':'validated','fire_time_end_s':float(field.times[-1]),'frames':len(field.times),
        'fds_elapsed_wall_s':float(elapsed.group(1)),'floors':stats,
        'peak_heat_release_kw':float(hrr[:,1].max()),
        'scenario_sha256':sha(snapshot),'deck_sha256':sha(RUN/'gaehwasan.fds'),
        'fields_sha256':field.meta['fields_sha256'],'out_sha256':sha(RUN/'gaehwasan.out'),
        'requested_slice_heights_m':field.meta['requested_slice_height_m'],
        'actual_slice_heights_m':field.meta['fire']['slice_height_m'],
        'passenger_results_unchanged':True,'passenger_simulation_run':False,
        'geometry_limits':field.meta.get('geometry_audit',{}).get('limits',[])}
    qa=ROOT/'output/qa/station_fire_20261029';qa.mkdir(parents=True,exist_ok=True)
    (qa/'validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    # Static plots of measured FDS fields, never a synthesized smoke overlay.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.colors import LogNorm
    from matplotlib.patches import Polygon as PlotPolygon
    font=qa/'malgun.ttf'
    if font.exists():font_manager.fontManager.addfont(str(font));plt.rcParams['font.family']='Malgun Gothic'
    plt.rcParams['axes.unicode_minus']=False
    fig=plt.figure(figsize=(12,9),layout='constrained');gs=fig.add_gridspec(2,2,height_ratios=[1,2])
    axes={'B2':fig.add_subplot(gs[0,:]),'B1':fig.add_subplot(gs[1,0]),'F1':fig.add_subplot(gs[1,1])}
    for floor in config['floors']:
        fid=floor['id'];ax=axes[fid]
        values=np.ma.masked_where(~masks[fid],np.maximum(field.data[fid+'_k'][-1],.001))
        image=ax.pcolormesh(field.data['x'],field.data['y'],values.T,shading='nearest',
            cmap='YlOrRd',norm=LogNorm(vmin=.001,vmax=max(1,max(s['max_extinction_per_m'] for s in stats.values()))))
        for ring in floor['polygons']:
            points=np.asarray(ring+[ring[0]]);ax.plot(points[:,0],points[:,1],color='#55616a',linewidth=.8)
        for obstacle in floor['obstacles']:
            if obstacle.get('render_as')=='building':
                ax.add_patch(PlotPolygon(obstacle['polygon'],facecolor='#c5cbcf',edgecolor='#6e7980',linewidth=.5))
        bounds=unary_union([Polygon(r) for r in floor['polygons']]).bounds
        ax.set_xlim(bounds[0]-2,bounds[2]+2);ax.set_ylim(bounds[1]-2,bounds[3]+2)
        ax.set_aspect('equal');ax.set_title(f'{fid} · 화재 600초의 소광계수');ax.set_xlabel('가로 위치 (m)');ax.set_ylabel('세로 위치 (m)')
        ax.set_facecolor('#edf0f2')
    colorbar=fig.colorbar(image,ax=list(axes.values()),label='소광계수 (1/m) · 로그 눈금 · 0.001 이하 최저색',shrink=.7)
    colorbar.set_ticks([.001,.01,.1,1]);colorbar.set_ticklabels(['0.001','0.01','0.1','1'])
    fig.suptitle('수정한 역사 지도 · FDS 화재·연기 계산 결과',fontsize=16)
    fig.savefig(qa/'smoke_600s.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for fid,(k,t) in curves.items():axes[0].plot(field.times,k,label=fid);axes[1].plot(field.times,t,label=fid)
    for ax,title,unit in zip(axes,['층별 최대 소광계수','층별 최대 온도'],['소광계수 (1/m)','온도 (°C)']):
        ax.set_title(title);ax.set_xlabel('화재 발생 후 시간 (초)');ax.set_ylabel(unit);ax.grid(alpha=.2);ax.legend()
    fig.savefig(qa/'floor_history.png',dpi=150);plt.close(fig)
    manifest.update(status='validated',validation_path='output/qa/station_fire_20261029/validation.json')
    (RUN/'run_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':validate()
