"""Post-rejection attribution; load the frozen fitted state, never refit."""
import json
import numpy as np
from free_arm_regions import ROOT, Source, Reconstruction, observations, summary, source_points, sha
from counterview_consistency import feasible, minimum_padding
from physical_release_comparison import launch_states
from hand_speed_derivative import metrics_for_state


def loaded():
    result=json.loads((ROOT/'free_arm_results.json').read_text())
    s=Source();x=result['solver']
    m=Reconstruction(s,np.array(x['root_translation_mm'])/1000,np.array(x['knot_displacements_mm'])/1000,np.array(x['segment_lengths_mm'])/1000)
    return s,m,result


def run():
    s,m,r=loaded();train,test=observations();groups={}
    for name,rows in [('camera17_at_fit_times',[x for x in test if x['camera']==17 and x['frame'] in (940,950,960)]),
                      ('early_other_times',[x for x in test if x['frame']<971 and x['frame'] not in (940,950,960)]),
                      ('release_window',[x for x in test if x['frame']>=971])]:
        groups[name]={label:summary(s,model,rows) for label,model in [('source',None),('reconstructed',m)]}
    lp={}
    for frame in [940,960]:
        for segment in ['l_hand','l_larm']:
            rows=[x for x in train+test if x['frame']==frame and x['segment']==segment]
            lp[f'{frame}_{segment}']={'cameras':[x['camera'] for x in rows],'feasibility':feasible(rows,s.cameras),'minimum_common_padding_px':minimum_padding(rows,s.cameras)}
    frames=np.arange(973,977);rotation0=s.hand(frames+1).as_matrix();rotation1=m.hand(frames)
    origin0=s.origins['l_hand'](frames+1);origin1=m.points(frames)[:,2]
    condition=next(x for x in s.observation['conditions'] if x['threshold']==80 and x['c3d_frame_offset']==1)
    ball=np.array([x['ball_rig_m'] for x in condition['positions']]);rate=float(s.observation['rate_hz']);controls={}
    for name,o,h in [('source',origin0,rotation0),('wrist_only',origin1,rotation0),('rotation_only',origin0,rotation1),('both',origin1,rotation1)]:
        state=launch_states(np.arange(4)/rate,ball[:4],o,h)['B']
        controls[name]=metrics_for_state(*state,s.gravity,ball[4:],rate,62.7*1.609344)
    result={'protocol_sha256':sha(ROOT/'FREE_ARM_DIAGNOSTIC_PROTOCOL.md'),'input_results_sha256':sha(ROOT/'free_arm_results.json'),
            'groups':groups,'region_lp':lp,'attribution_controls':controls,'refitted':False,'rejection_reversed':False,
            'maximum_hand_transport_angle_deg':float(np.degrees(max(__import__('scipy').spatial.transform.Rotation.from_matrix(a@b.T).magnitude() for a,b in zip(rotation1,rotation0)))),
            'actual_athlete_gain_kmh':None,'force_work_change':None}
    (ROOT/'free_arm_diagnostics.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'groups':{k:{n:x['mean_outside_px'] for n,x in v.items()} for k,v in groups.items()},'lp':lp,
                      'controls':{k:v['mean_absolute_speed_discrepancy_kmh'] for k,v in controls.items()},'angle':result['maximum_hand_transport_angle_deg']}))
    return result


if __name__=='__main__':run()
