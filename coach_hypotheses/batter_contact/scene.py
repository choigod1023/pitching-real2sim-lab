"""Separate OBP pitching/hitting records in an explicitly constructed matchup."""
from contextlib import ExitStack
import json
from pathlib import Path
import zipfile

import ezc3d
import numpy as np
import pandas as pd

from ..observed_motion.overlay import JOINTS, EDGES, interpolate, SOURCE_HASHES
from ..temporal_quality.model import digest,save
from .geometry import ballistic_initial,sweep

HITTER_JOINTS=('lhjc','rhjc','lsjc','rsjc','lejc','rejc','lwjc','rwjc','lkjc','rkjc','lajc','rajc')
HITTER_EDGES=(('lhjc','rhjc'),('lsjc','rsjc'),('lhjc','lsjc'),('rhjc','rsjc'),('lsjc','lejc'),
 ('lejc','lwjc'),('rsjc','rejc'),('rejc','rwjc'),('lhjc','lkjc'),('lkjc','lajc'),('rhjc','rkjc'),('rkjc','rajc'))
ROTATION=np.diag([-1.,-1.,1.])
ARCHIVE_SHA='9cc2bdc94f5f5a6c5d540f4abe459caae2c5164cd53f6a6fd23f4b2b7283bcb3'
C3D_SHA='083980649dfb4c3069a4f10baff57f156fcc4fb1d52decdba24f487e2ebc18cf'


def selected_table(path,key,value,zipped=False):
    with ExitStack() as stack:
        if zipped:
            archive=stack.enter_context(zipfile.ZipFile(path));source=stack.enter_context(archive.open('landmarks.csv'))
        else:source=path
        found=[]
        for chunk in pd.read_csv(source,chunksize=20000):
            selected=chunk[chunk[key].eq(value)]
            if len(selected):found.append(selected)
        if not found:raise ValueError('missing_selected_record')
        result=pd.concat(found).sort_values('time').reset_index(drop=True)
    if result.time.duplicated().any():raise ValueError('duplicate_record_time')
    return result


def canonical_time(frame):
    time=np.arange(len(frame))/360.
    if len(time)<2 or np.max(abs(time-frame.time.to_numpy()))>.000051:raise ValueError('invalid_360hz_grid')
    return time


def choose_hitter(metadata):
    selected=metadata[(metadata.hitter_side=='R')&(metadata.bat_length_in>0)].copy()
    if selected.empty:raise ValueError('no_eligible_hitter')
    selected['numeric_user']=selected.user.astype(int)
    selected['numeric_session']=selected.session.astype(int)
    selected['numeric_swing']=selected.session_swing.str.split('_').str[-1].astype(int)
    return selected.sort_values(['numeric_user','numeric_session','numeric_swing']).iloc[0]


def load_scene(obp_dir, external, output):
    obp_dir,external,output=map(Path,(obp_dir,external,output));output.mkdir(parents=True,exist_ok=True)
    hitting_zip=external/'hitting_landmarks.zip';raw_zip=external/'hitting_c3d.zip'
    if digest(hitting_zip)!=ARCHIVE_SHA or digest(raw_zip)!=C3D_SHA:raise ValueError('upstream_archive_digest_mismatch')
    pitch_dir=obp_dir/'baseball_pitching/data';hitting_dir=obp_dir/'baseball_hitting/data'
    for name in ('metadata.csv','full_sig/landmarks.csv'):
        if digest(pitch_dir/name)!=SOURCE_HASHES[name]:raise ValueError('changed_pitcher_source')
    meta=pd.read_csv(pitch_dir/'metadata.csv').set_index('session_pitch').loc['1615_1']
    pitcher=selected_table(pitch_dir/'full_sig/landmarks.csv','session_pitch','1615_1')
    hm=pd.read_csv(hitting_dir/'metadata.csv',dtype={'user':str,'session':str})
    first=choose_hitter(hm)
    if first.session_swing!='103_1':raise ValueError('amended_first_hitter_changed')
    hitter=selected_table(hitting_zip,'session_swing',first.session_swing,True)
    pt,ht=canonical_time(pitcher),canonical_time(hitter)
    if pitcher.BR_time.nunique()!=1 or hitter.contact_time.nunique()!=1:raise ValueError('nonconstant_event')
    xyz=np.stack([pitcher[[n+'_'+a for a in 'xyz']].to_numpy(float) for n in JOINTS],axis=1)
    hips=(xyz[:,JOINTS.index('rear_hip')]+xyz[:,JOINTS.index('lead_hip')])/2
    origin=interpolate(pt,hips,float(pitcher.fp_10_time.iloc[0]));origin[2]=0
    xyz=(xyz-origin)@ROTATION.T+np.array([18.44,0,0])
    hxyz=np.stack([hitter[[n+'_'+a for a in 'xyz']].to_numpy(float) for n in HITTER_JOINTS],axis=1)
    grip=hitter[['blast_hand_'+a for a in 'xyz']].to_numpy(float)
    sweet=hitter[['sweet_spot_'+a for a in 'xyz']].to_numpy(float)
    if not all(np.isfinite(v).all() for v in (xyz,hxyz,grip,sweet)):raise ValueError('missing_source_pose_no_imputation')
    br=float(pitcher.BR_time.iloc[0]);contact=float(hitter.contact_time.iloc[0])
    release=interpolate(pt,xyz[:,JOINTS.index('hand_jc')],br)
    destination=interpolate(ht,sweet,contact);speed=float(meta.pitch_speed_mph)*1.609344
    velocity,arrival=ballistic_initial(release,destination,speed/3.6)
    raw_candidates=[]
    with zipfile.ZipFile(raw_zip) as z:
        prefix=f'c3d/{first.user}/{first.user}_{first.session}_'
        for name in sorted(z.namelist()):
            if not name.startswith(prefix) or name.endswith('model.c3d'):continue
            path=output/Path(name).name;path.write_bytes(z.read(name));raw=ezc3d.c3d(str(path))
            point=raw['parameters']['POINT'];labels=point['LABELS']['value']
            if point['UNITS']['value']!=['m'] or point['RATE']['value'][0]!=360:
                raise ValueError('hitting_raw_units_or_rate_mismatch')
            if raw['data']['points'].shape[-1]!=len(hitter):continue
            wrist=(raw['data']['points'][:3,labels.index('RWRA')].T+raw['data']['points'][:3,labels.index('RWRB')].T)/2
            error=np.linalg.norm(wrist-hitter[['rwjc_'+a for a in 'xyz']].to_numpy(),axis=1)*1000
            raw_candidates.append((float(np.median(error)),path))
    if not raw_candidates:raise ValueError('no_raw_frame_correspondence_candidate')
    wrist_error,raw_path=min(raw_candidates,key=lambda pair:(pair[0],pair[1].name))
    scene=dict(pitch_time=pt,pitch_xyz=xyz,br=br,hitter_time=ht,hitter_xyz=hxyz,
       bat=dict(time=ht,grip=grip,sweet=sweet,contact=contact),release=release,velocity=velocity,
       arrival=arrival,speed_kmh=speed,hitter_id=first.user,swing_id=first.session_swing)
    audit=dict(pitch='1615_1',pitcher='830',swing=first.session_swing,hitter=first.user,same_real_matchup=False,
       protocol_commit='3daae1d263c995fe6c86823aba85fe692675a38c',
       source_hashes={str(p.resolve()):digest(p) for p in (hitting_zip,raw_zip,pitch_dir/'metadata.csv',
             pitch_dir/'full_sig/landmarks.csv',hitting_dir/'metadata.csv',raw_path)},
       raw_hitting_unit='m',raw_hitting_rate_hz=360,
       hitting_raw_match='unique_frame_count_and_wrist_geometric_correspondence_not_explicit_trial_id',
       identity_amendment_commit='ad9fec6',
       hitting_raw_processed_wrist_median_mm=wrist_error,
       hitting_raw_correspondence_candidates=[dict(median_wrist_mm=error,name=path.name) for error,path in raw_candidates],
       recorded_pitch_speed_used_as_initial_assumption_kmh=speed,flight_time_ms=arrival*1000,
       release_origin='processed_hand_center_proxy_not_observed_ball',
       ball_destination='forced_through_provided_sweet_spot_at_contact_not_observed_ball',
       pitcher_rotation=ROTATION.tolist(),pitcher_translation=[18.44,0,0],
       pitcher_translation_reference='FP10_hip_scene_anchor_not_calibrated_rubber',
       pitcher_pose_change_mm=0.,pitcher_angle_change_deg=0.,force_change=None,work_change=None,
       spin=None,drag=None,postimpact_exit_velocity=None,postimpact_launch_angle=None,
       actual_athlete_improvement=False,data_license='CC BY-NC-SA 4.0')
    return scene,audit


def run_geometry(scene):
    args=(scene['bat'],scene['release'],scene['velocity'],scene['arrival'])
    nominal=sweep(*args)
    convergence=[]
    for step in (.0002,.0001,.00005):
        value=sweep(*args,step=step)
        convergence.append(dict(step_ms=step*1000,intervals_ms=value['grid_near_intervals_ms'],
            selected_cases=[r for r in value['cases'] if r['delay_ms'] in (0,12)]))
    sensitivity=[]
    for radius in (.028,.033,.038):
        for offset in (-.01,0,.01):
            value=sweep(*args,bat_radius=radius,z_offset=offset)
            sensitivity.append(dict(bat_radius_mm=radius*1000,z_offset_mm=offset*1000,
                intervals_ms=value['grid_near_intervals_ms'],selected_cases=[r for r in value['cases'] if r['delay_ms'] in (0,12)]))
    return dict(nominal=nominal,convergence=convergence,engineering_sensitivity=sensitivity,
       baseline_near_by_construction=True,matched_player_validation=False,actual_whiff_probability=None,
       same_force_and_work_validated=False,athlete_improvement=False)
