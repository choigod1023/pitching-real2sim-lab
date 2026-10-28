"""Compare overlapping and visibly separated training windows on later cameras."""
import json,hashlib
import numpy as np
from pixel_launch_validation import ROOT,fit,project,trajectory
from physical_release_comparison import free_flight

def run():
    lab=ROOT.parent
    source=lab/'speed-journey/artifacts/release-observation/candidate_audit.json'
    later_path=lab/'ball-trajectory-fit/artifacts/trajectory-fit/trajectory_fit.json'
    a=json.loads(source.read_text());later=json.loads(later_path.read_text())
    cal=lab/'openbiomechanics/computer_vision/calibration/results'
    intr=json.loads((cal/'calibration_results.json').read_text())['calibrations']
    rig=json.loads((cal/'optitrack_rig_provisional.json').read_text())['camera_poses']
    base=json.loads((ROOT/'artifacts/pixel-launch/results.json').read_text());g=np.array(base['gravity'])
    cameras={}
    for n in (19,20,21,22):
        k=intr[f'optitrack_{n}'];e=rig[str(n)]
        if any(k['distortion']):raise ValueError('Unsupported distortion')
        cameras[n]=(np.array([[k['fx_px'],0,k['cx_px']],[0,k['fy_px'],k['cy_px']],[0,0,1.]]),np.array(e['reference_to_camera_rotation_matrix']),np.array(e['reference_to_camera_translation_mm'])/1000)
    # Duplicate detections from the old threshold-pair loops must not be counted repeatedly.
    unique={}
    for d in later['detections']:
        if d['frame'] in (980,981,985):
            key=(d['camera'],d['frame'],d['threshold'],d['mode'])
            if key in unique:assert unique[key]==d
            unique[key]=d
    rows=[]
    for threshold in (60,80,100):
        tri={r['frame']:np.array(r['position_camera19_mm'])/1000 for r in a['conditional_triangulation'] if r['threshold']==threshold}
        det={(r['camera'],r['frame']):r for r in a['detections'] if r['threshold']==threshold}
        for window,frames in [('overlap',np.arange(973,977)),('separated',np.arange(977,980))]:
            t=(frames-frames[-1])/360
            points=np.array([tri[f] for f in frames]);pixels=np.array([[det[c,f]['pixel'] for f in frames] for c in (20,21)])
            for mode in ('C','G','P','R'):
                state=fit(t,points,pixels,[cameras[20],cameras[21]],g,mode)
                future,_=free_flight(state[:3],state[3:],g,substeps=2,nframes=int(985-frames[-1]))
                selection=np.array([980,981,985])-frames[-1]-1
                predicted=future[selection]
                assert np.max(np.abs(predicted-trajectory(state,(np.array([980,981,985])-frames[-1])/360,g)))<1e-10
                checks=[]
                for d in unique.values():
                    j=[980,981,985].index(d['frame']);uv=project(predicted[[j]],cameras[d['camera']])[0]
                    checks.append(dict(camera=d['camera'],frame=d['frame'],threshold=d['threshold'],extraction=d['mode'],status=d['status'],error_px=float(np.linalg.norm(uv-d['pixel'])) if d['status']=='candidate' else None))
                summaries=[]
                for cam in (19,22):
                    for extraction in ('component','local_ellipse'):
                        checks_subset=[x for x in checks if x['camera']==cam and x['extraction']==extraction]
                        err=[x['error_px'] for x in checks_subset if x['error_px'] is not None]
                        summaries.append(dict(camera=cam,extraction=extraction,n=len(err),failures=len(checks_subset)-len(err),rmse_px=float(np.sqrt(np.mean(np.square(err))))))
                # Same nonuniform-time linear summary for predicted and held-out triangulated positions.
                design=np.column_stack([np.ones(3),(np.array([980,981,985])-980)/360])
                speed=float(np.linalg.norm(np.linalg.lstsq(design,predicted,rcond=None)[0][1])*3.6)
                optical=[]
                for case in later['cases']:
                    if 'linear_visible_primary' in case:
                        value=case['linear_visible_primary']['midpoint_speed_kmh']
                        optical.append(dict(extraction=case['mode'],thresholds=case['thresholds'],observed_speed_kmh=value,absolute_difference_kmh=abs(speed-value)))
                rows.append(dict(source_threshold=threshold,window=window,model=mode,training_frames=frames.tolist(),speed_kmh=speed,radar_difference_kmh=abs(speed-62.7*1.609344),summaries=summaries,checks=checks,optical_comparisons=optical))
    report=dict(scope='Later cameras and frames excluded from fitting; previously inspected development data, not truth. No hand-to-ball causal prediction.',results=rows,
        sources_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [source,later_path,ROOT/'FLIGHT_WINDOW_PROTOCOL.md',cal/'calibration_results.json',cal/'optitrack_rig_provisional.json',ROOT/'artifacts/pixel-launch/results.json']})
    (ROOT/'artifacts/pixel-launch/window_results.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
    for r in rows:
        if r['source_threshold']==80:print(r['window'],r['model'],round(r['speed_kmh'],4),[(s['camera'],s['extraction'],round(s['rmse_px'],3)) for s in r['summaries']])
    return report

if __name__=='__main__':run()
