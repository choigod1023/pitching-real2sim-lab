"""Post-failure shape/scale attribution; never apply a speed multiplier."""
import json
import numpy as np
from prime_cube_correction import cube_chart,rigid_registration
from prime_cube_audit import ROOT,VERTICES


def run():
    r=json.loads((ROOT/'prime_cube_results.json').read_text());chart=cube_chart();rows={}
    for name,v in r['pairs'].items():
        x=np.array([chart[k] for k in VERTICES]);y=np.array([v['points_m'][k] for k in VERTICES]);rot,t=rigid_registration(x,y)
        xc=x-x.mean(0);yc=y-y.mean(0)
        scale=float(np.sum((xc@rot.T)*yc)/np.sum(xc*xc))
        fit=scale*(x@rot.T)+y.mean(0)-scale*rot@x.mean(0)
        rows[name]={'diagnostic_similarity_scale_expected_to_triangulated':scale,
                    'rigid_shape_rmse_mm':float(np.sqrt(np.mean(np.sum((x@rot.T+t-y)**2,axis=1)))*1000),
                    'similarity_shape_rmse_mm':float(np.sqrt(np.mean(np.sum((fit-y)**2,axis=1)))*1000),
                    'applied_to_speed_or_cameras':False}
    result={'scope':'Post-failure shape/scale attribution, not another correction or selected physical calibration','pairs':rows}
    (ROOT/'prime_cube_scale_diagnostic.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    return result


if __name__=='__main__':run()
