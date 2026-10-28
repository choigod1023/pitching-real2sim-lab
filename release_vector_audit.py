"""Prespecified velocity-vector decomposition, never fitted to radar outcomes."""
import json
from pathlib import Path
import ezc3d
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import butter, sosfiltfilt
from scipy.spatial.transform import Rotation, RotationSpline
from trace_error_audit import ObservedPitch, sha
from speed_path import retarget_velocity, rotation_velocity
from ball_release import hand_frame


def vector_metrics(v):
    v=np.asarray(v,float)
    if v.shape!=(3,) or not np.isfinite(v).all():raise ValueError('finite velocity vector required')
    return dict(velocity_m_s=v.tolist(),speed_kmh=float(np.linalg.norm(v)*3.6),
        elevation_deg=float(np.degrees(np.arctan2(v[2],np.hypot(v[0],v[1])))),
        azimuth_from_home_deg=float(np.degrees(np.arctan2(v[1],v[0]))))


def field_frame(p,v,origin_xy):
    """Documented lab +x home, +y first base, +z up -> flight right/forward/up.

    Fixed proper rotation, not reoriented to each candidate's own velocity.
    Height remains in laboratory z; this is not a surveyed plate transform.
    """
    p,v=np.asarray(p,float),np.asarray(v,float)
    if p.shape!=(3,) or v.shape!=(3,) or not np.isfinite(np.r_[p,v,origin_xy]).all():
        raise ValueError('finite release coordinates required')
    basis=np.array([[0,-1,0],[1,0,0],[0,0,1.]])
    return basis@(p-np.r_[origin_xy,0]),basis@v


def run():
    out=Path('artifacts/release-vector');out.mkdir(parents=True,exist_ok=True)
    rows=[];sources=[]
    for pitch in ['3034_1','3034_2','3034_3','3034_4','3034_5']:
        obs=ObservedPitch(pitch);sources.append(obs.ledger)
        c=ezc3d.c3d(obs.ledger['source_paths']['c3d'])
        labels=[s.strip() for s in c['parameters']['POINT']['LABELS']['value']]
        scale={'m':1,'cm':.01,'mm':.001}[obs.ledger['raw_point_units']]
        raw=[c['data']['points'][:3,labels.index(n),:].T*scale for n in ['RFIN','RWRA','RWRB']]
        ts=np.arange(len(raw[0]))/obs.ledger['rate_hz'];rate=obs.ledger['rate_hz']
        for cutoff in [0,20]:
            marks=raw if cutoff==0 else [sosfiltfilt(butter(4,cutoff,fs=rate,output='sos'),m,axis=0) for m in raw]
            _,frames=hand_frame(*marks);rot=RotationSpline(ts,Rotation.from_matrix(frames))
            fin=CubicSpline(ts,marks[0]);mid=CubicSpline(ts,(marks[1]+marks[2])/2)
            for shift in [-2,-1,0,1,2]:
                t=obs.b+shift/rate;points=obs.points(t);vel=obs.points(t,1)
                wrist=retarget_velocity(points,vel,obs.lengths)
                rotation=rotation_velocity(rot,t,np.array([.15,0,0]))
                vectors={'rfin_marker':fin(t,1),'wrist_marker_mid':mid(t,1),
                    'processed_wrist':vel[2],'fixed_length_wrist':wrist,
                    'rotation_offset_component':rotation,'assumed_ball':wrist+rotation}
                rows.append(dict(pitch=pitch,cutoff_hz=cutoff,br_shift_frames=shift,
                    shifts_ms=shift/rate*1000,metrics={k:vector_metrics(v) for k,v in vectors.items()},
                    wrist_height_m=float(points[2,2]),rfin_height_m=float(fin(t)[2]),
                    assumed_ball_height_m=float(points[2,2]+(rot(t).as_matrix()@np.array([.15,0,0]))[2])))
    report=dict(protocol='five ordered pitches; raw and20Hz; BR -2/-1/0/1/2 frames; 15cm fixed grip; no selection by target speed or target point',
        coordinates='OBP pitching README: +x toward home, +y toward first base, +z up',
        readme_sha256=sha(Path(__file__).parent.parent/'openbiomechanics/baseball_pitching/README.md'),
        sources=sources,rows=rows,scope='Marker and assumed rigid-ball velocities, not measured ball velocities; BR physical detection unconfirmed')
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    for row in rows:
        if row['cutoff_hz']==20 and row['br_shift_frames']==0:
            print(row['pitch'],{k:round(v['elevation_deg'],3) for k,v in row['metrics'].items()})
    return report


if __name__=='__main__':run()
