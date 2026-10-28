import json
import numpy as np
from noise_structure_audit import OUT

def main():
    records=json.loads((OUT/'report.json').read_text())['records'];summary={};factorial=[]
    for noise in ['original','rounding_bins']:
        for structure in ['fixed_origin','measured_mobile_origin']:
            rows=sorted([r for r in records if r['noise']==noise and r['structure']==structure],key=lambda r:r['pitch'])
            if len(rows)!=5:raise ValueError('Expected five pitches per cell')
            first=rows[0]
            out={key:float(np.mean([r[key] for r in rows])) for key in ['arm_rmse_mm','root_force_rms_n','root_torque_rms_nm','internal_positive_work_j','root_residual_positive_work_j','added_origin_positive_work_j']}
            out['speed_mae_kmh']=float(np.mean([abs(r['signed_speed_error_kmh']) for r in rows]))
            out['change_from_first_mae_kmh']=float(np.mean([abs((r['speed_kmh']-first['speed_kmh'])-(r['measured_speed_kmh']-first['measured_speed_kmh'])) for r in rows[1:]]))
            out['no_change_mae_kmh']=float(np.mean([abs(r['measured_speed_kmh']-first['measured_speed_kmh']) for r in rows[1:]]))
            summary[noise+'__'+structure]=out
    for pitch in sorted({r['pitch'] for r in records}):
        cell={(r['noise'],r['structure']):r for r in records if r['pitch']==pitch}
        a=cell['original','fixed_origin'];b=cell['rounding_bins','fixed_origin'];c=cell['original','measured_mobile_origin'];d=cell['rounding_bins','measured_mobile_origin']
        result={'pitch':pitch}
        for metric in ['speed_kmh','root_force_rms_n','arm_rmse_mm']:
            result[metric]=dict(rounding_only=b[metric]-a[metric],structure_only=c[metric]-a[metric],
                interaction=d[metric]-b[metric]-c[metric]+a[metric],combined=d[metric]-a[metric])
        factorial.append(result)
    z=np.load(OUT/'3034_4_overlay.npz');names=list(z['names'])
    low=[names.index(n) for n in names if n.startswith(('rear_','lead_'))]
    delta=z['rounding_bins__measured_mobile_origin'][:,low]-z['original__fixed_origin'][:,low]
    result=dict(summary=summary,factorial=factorial,example_lower_body_change_rmse_mm=float(np.sqrt(np.mean(np.sum(delta**2,axis=2)))*1000),
        ability_improvement_demonstrated=False,ready_for_same_capacity_optimization=False,
        reason='Observed shoulder translation is prescribed and has new unconstrained actuator forces. Ground residual remains. Speed change prediction must also beat no-change. Reconstruction improvement is not an intervention effect.')
    (OUT/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
