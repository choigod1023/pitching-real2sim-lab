import hashlib,json
from pathlib import Path
import ezc3d,numpy as np
from hand_speed_derivative import ROOT,SOURCE,analytic_state,numerical_rotations,metrics_for_state,improves

def run():
    base=json.loads((ROOT/'results.json').read_text());ob=json.loads(SOURCE.read_text(encoding='utf-8'));ledger=ob['source_sha256'];alignment=json.loads(Path(ledger['alignment']['path']).read_text());q=np.array(alignment['rotation']);c=ezc3d.c3d(ledger['body']['path']);idx=c['parameters']['ROTATION']['LABELS']['value'].index('l_hand_4X4');rr=c['data']['rotations'][:3,:3,idx,:];frames=np.arange(973,977);rate=ob['rate_hz'];times=(frames-973)/rate;rows=[]
    for row,previous in zip(ob['conditions'],base['conditions']):
        assert (row['threshold'],row['c3d_frame_offset'])==(previous['threshold'],previous['offset'])
        ball=np.array([v['ball_rig_m'] for v in row['positions']]);origin=np.array([v['hand_origin_rig_m'] for v in row['positions']]);r,_=numerical_rotations(np.einsum('ij,jkn->nik',q,rr[:,:,frames+row['c3d_frame_offset']]));p,v,detail=analytic_state(times,ball[:4],origin[:4],r,2,2);m=metrics_for_state(p,v,np.array(base['gravity_rig_m_s2']),ball[4:],rate,base['source_label_kmh']);rows.append({'threshold':row['threshold'],'offset':row['c3d_frame_offset'],'metrics':m,'detail':detail,'launch_position_change_from_B_mm':float(np.linalg.norm(p-np.array(previous['metrics']['B']['launch_position_m']))*1000),'beats_B_all_metrics':improves(m,previous['metrics']['B']),'beats_C_all_metrics':improves(m,previous['metrics']['C'])})
    primary=next(x for x in rows if x['threshold']==80 and x['offset']==1);out={'scope':'Post-primary exploratory relative-curvature diagnostic, same already inspected development evaluation','primary_frozen_D2_rejection_unchanged':True,'primary_condition_result':primary,'conditions':rows,'scientific_stage_passed':False,'causal_intervention_validated':False,'protocol_sha256':hashlib.sha256(Path('RELATIVE_CURVATURE_PROTOCOL.md').read_bytes()).hexdigest()};(ROOT/'relative-curvature-results.json').write_text(json.dumps(out,indent=2));print(json.dumps({'primary':primary,'conditions_beating_B_all_metrics':sum(r['beats_B_all_metrics'] for r in rows),'conditions_beating_C_all_metrics':sum(r['beats_C_all_metrics'] for r in rows)},indent=2));return out
if __name__=='__main__':run()
