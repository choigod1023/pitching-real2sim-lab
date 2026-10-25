"""Audit actual decoded presentation times and conditional stereo sensitivity.

Container timestamps establish media cadence, not sensor exposure synchronization.
No camera parameter is refitted and no radar value is used.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re
import subprocess
import numpy as np
from PIL import Image
from ball_trajectory_fit import extract, seed, load_cameras, triangulate, fit_trajectory


def timestamp_summary(pts, timebase):
    pts = np.asarray(pts)
    if pts.ndim != 1 or len(pts) < 2 or not np.isfinite(timebase) or timebase <= 0:
        raise ValueError('Need two or more timestamps and positive timebase')
    if not np.issubdtype(pts.dtype,np.integer):raise ValueError('PTS must be integer ticks')
    pts=pts.astype(np.int64)
    delta = np.diff(pts)
    return {'frames': len(pts), 'first_pts': int(pts[0]), 'last_pts': int(pts[-1]),
            'timebase_seconds': timebase, 'strictly_increasing': bool(np.all(delta > 0)),
            'interval_ticks_counts': {str(int(k)): int(v) for k,v in zip(*np.unique(delta,return_counts=True))},
            'interval_min_ms': float(delta.min()*timebase*1000),
            'interval_max_ms': float(delta.max()*timebase*1000),
            'interval_median_ms': float(np.median(delta)*timebase*1000),
            'intervals_above_1_5_median': int(np.sum(delta > 1.5*np.median(delta))),
            'duplicate_pts': int(np.sum(delta == 0)), 'pts': pts.tolist()}


def decode_timestamps(path, ffmpeg):
    result = subprocess.run([str(ffmpeg), '-hide_banner', '-i', str(path), '-map', '0:v:0',
                             '-vf', 'showinfo', '-fps_mode', 'passthrough', '-f', 'null', '-'],
                            capture_output=True, text=True, encoding='utf-8', errors='replace', check=True)
    tb = re.search(r'config in time_base:\s*(\d+)/(\d+)',result.stderr)
    frames = re.findall(r'n:\s*(\d+)\s+pts:\s*(-?\d+)\s+pts_time:', result.stderr)
    if not tb or not frames: raise ValueError('Missing decoded timestamp metadata')
    if [int(n) for n,_ in frames] != list(range(len(frames))):
        raise ValueError('Decoded frame indexes are not contiguous')
    row = timestamp_summary([int(p) for _,p in frames], int(tb[1])/int(tb[2]))
    version=subprocess.run([str(ffmpeg),'-version'],capture_output=True,text=True,check=True).stdout.splitlines()[0]
    row.update(sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size,
               ffmpeg_version=version)
    return row


def fundamental(cameras):
    (k1,p1),(k2,p2) = cameras
    r = p2[:,:3] @ p1[:,:3].T
    t = p2[:,3] - r @ p1[:,3]
    tx = np.array([[0,-t[2],t[1]],[t[2],0,-t[0]],[-t[1],t[0],0.]])
    return np.linalg.inv(k2).T @ tx @ r @ np.linalg.inv(k1)


def epipolar_distance(pair, cameras):
    """RMS of point-to-epipolar-line distances in both undistorted images."""
    a,b = (np.r_[p,1.] for p in pair)
    f = fundamental(cameras); l2=f@a; l1=f.T@b
    den = np.array([l1[:2]@l1[:2],l2[:2]@l2[:2]])
    if np.any(den<=1e-24): raise ValueError('Degenerate epipolar geometry')
    return float(abs(b@f@a)*np.sqrt(np.mean(1/den)))


def shifted_track_audit(tracks, cameras, frames, shifts=(-2,-1,0,1,2)):
    """Fixed anchor frames for comparable shifts. Flags never silently discarded."""
    output=[]
    for shift in shifts:
        rows=[]
        for frame in frames:
            one=tracks.get((19,int(frame))); two=tracks.get((22,int(frame)+shift))
            if one is None or two is None or one.get('pixel') is None or two.get('pixel') is None:
                rows.append({'frame':int(frame),'status':'missing'}); continue
            pair=[one['pixel'],two['pixel']]
            x,res,dep=triangulate(pair,cameras)
            flags=[r.get('status','unknown') for r in (one,two)]
            visible=all(r.get('visible',False) for r in (one,two))
            rows.append({'frame':int(frame),'camera22_frame':int(frame)+shift,
                         'status':'conditional','visible_pair':visible,'flags':flags,
                         'epipolar_rms_px':epipolar_distance(pair,cameras),
                         'position_mm':x.tolist(),'reprojection_px':res,'positive_depth':min(dep)>0})
        good=[r for r in rows if r['status']=='conditional']
        row={'shift_camera22_frames':shift,'shift_ms':shift/360*1000,'rows':rows,
             'not_a_sync_estimate':True,'all_anchor_frames_available':len(good)==len(frames)}
        if len(good)>=2:
            row['epipolar_median_px']=float(np.median([r['epipolar_rms_px'] for r in good]))
            row['fit']=fit_trajectory(np.array([r['frame'] for r in good])/360,
                                      np.array([r['position_mm'] for r in good]))
        output.append(row)
    return output


def extended_track_audit(path, calibration, output):
    data=json.loads(path.read_text(encoding='utf-8'))
    detections=data['detections'];cameras=load_cameras(calibration)
    intr=json.loads((calibration/'calibration_results.json').read_text())['calibrations']
    groups=sorted({(r['method'],r['threshold']) for r in detections})
    cases=[]
    for method,threshold in groups:
        rows=[r for r in detections if r['method']==method and r['threshold']==threshold]
        tracks={(r['camera'],r['frame']):dict(r,visible=r['status']=='candidate') for r in rows}
        shifts=shifted_track_audit(tracks,cameras,list(range(987,998)))
        coverage=[]
        for c in (19,22):
            cal=intr[f'optitrack_{c}']
            values=[]
            for r in rows:
                if r['camera']!=c or r['pixel'] is None or r['status']!='candidate':continue
                u,v=r['pixel'];radius=np.hypot((u-cal['cx_px'])/cal['fx_px'],(v-cal['cy_px'])/cal['fy_px'])
                values.append([r['frame'],float(radius)])
            # The calibration's max radius is a coarse envelope, not a spatial coverage guarantee.
            limit=cal['coverage']['observed_max_normalized_radius']
            coverage.append({'camera':c,'normalized_radii_by_frame':values,
                             'checkerboard_observed_radius_limit':limit,
                             'frames_beyond_observed_radius':[f for f,radius in values if radius>limit]})
        cases.append({'method':method,'threshold':threshold,'shifts':shifts,'coverage':coverage})
    artifact={'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'cases':cases,
              'anchors':list(range(987,998)),'selected_shift':None,
              'reason':'Engineering offset stress only; not an independently estimated timing correction.'}
    representative={(r['camera'],r['frame']):dict(r,visible=r['status']=='candidate') for r in detections
                    if r['method']=='weighted' and r['threshold']==20}
    fold_cases=[]
    if representative:
        for i,p19 in enumerate(intr['optitrack_19']['cross_validation']['fold_parameters']):
          for j,p22 in enumerate(intr['optitrack_22']['cross_validation']['fold_parameters']):
            modified=[]
            for parameters,(_,pose) in zip((p19,p22),cameras):
                fx,fy,cx,cy,k1,k2=parameters
                if k1!=0 or k2!=0:raise ValueError('Fold distortion requires explicit undistortion')
                modified.append((np.array([[fx,0,cx],[0,fy,cy],[0,0,1.]]),pose))
            result=shifted_track_audit(representative,modified,list(range(987,998)),[0])[0]
            fold_cases.append({'camera19_fold':i,'camera22_fold':j,'fit':result['fit'],
                               'epipolar_median_px':result['epipolar_median_px']})
    artifact['intrinsic_model_selection_audit']={str(c):{
        'selected_model':intr[f'optitrack_{c}']['model'],
        'candidate_models':intr[f'optitrack_{c}']['candidate_models'],
        'decision':'Retain upstream simplest model within heldout tolerance; radial changes require joint extrinsics refit and independent validation.'}
        for c in (19,22)}
    artifact['intrinsic_fold_stress']=fold_cases
    artifact['fold_stress_interpretation']='Empirical calibration-fit variation with extrinsics fixed; not joint recalibration or confidence interval.'
    (output/'extended_track_camera_audit.json').write_text(json.dumps(artifact,indent=2),encoding='utf-8')
    for c in cases:
        if c['method']=='weighted' and c['threshold']==20:
            print(json.dumps({'representative':[{k:v for k,v in r.items() if k!='rows'} for r in c['shifts']], 'coverage':c['coverage']}))
    return artifact


def run(source, calibration, ffmpeg, output, tracks_path=None):
    output.mkdir(parents=True,exist_ok=True)
    timing={str(c):decode_timestamps(source/f'camera{c}.mp4',ffmpeg) for c in (15,19,22)}
    timing_equal=all(timing[c]['pts']==timing['19']['pts'] and
                     timing[c]['timebase_seconds']==timing['19']['timebase_seconds'] for c in timing)
    cameras=load_cameras(calibration)
    tracks={}
    for c,threshold in ((19,100),(22,70)):
        for frame in range(978,988):
            path=source/f'camera{c}_frame{frame}.png'
            if not path.exists():continue
            r=extract(Image.open(path),seed(c,frame),threshold,'component')
            # Existing image-reviewed window only; outer frames are explicitly unreviewed.
            r['visible']=frame in (980,981,985) and r['status']=='candidate'
            tracks[c,frame]=r
    offsets=shifted_track_audit(tracks,cameras,[980,981,985])
    base=next(r['fit']['midpoint_speed_kmh'] for r in offsets if r['shift_camera22_frames']==0)
    scale=[{'scale':s,'speed_kmh':base*s,'change_kmh':base*(s-1)} for s in (.99,.995,1.,1.005,1.01)]
    rig=json.loads((calibration/'optitrack_rig_provisional.json').read_text())
    intrinsics=json.loads((calibration/'calibration_results.json').read_text())
    artifact={'timing':timing,'identical_container_timestamps':timing_equal,
              'sensor_sync_proven':False,'offset_stress':offsets,
              'uniform_scale_stress':scale,'scale_for_1kmh_fraction':1/base,
              'scale_stress_is_uncertainty_interval':False,
              'calibration_status':rig['status'], 'pair19_22':rig['accepted_pairs']['19_to_22'],
              'intrinsics':{str(c):intrinsics['calibrations'][f'optitrack_{c}'] for c in (15,19,22)},
              'calibration_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                  [calibration/'calibration_results.json',calibration/'optitrack_rig_provisional.json']}}
    (output/'camera_audit.json').write_text(json.dumps(artifact,indent=2),encoding='utf-8')
    if tracks_path is not None:extended_track_audit(tracks_path,calibration,output)
    print(json.dumps({'timing':{c:{k:v for k,v in r.items() if k!='pts'} for c,r in timing.items()},
                      'equal':timing_equal,'offsets':[{k:v for k,v in r.items() if k!='rows'} for r in offsets]}))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for field in ('source','calibration','ffmpeg','output'):parser.add_argument(field,type=Path)
    parser.add_argument('--tracks',type=Path)
    a=parser.parse_args();run(a.source,a.calibration,a.ffmpeg,a.output,a.tracks)
