"""3D feasibility of approximate image regions; never exact joint inference."""
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
from prime17_hand_alignment import Source, project, box_residual, sha

ROOT = Path(__file__).resolve().parent


def region_inequalities(box, camera):
    b = np.asarray(box, float); k, r, t = camera
    if b.shape != (4,) or not np.isfinite(b).all() or np.any(b[:2] > b[2:]):
        raise ValueError('Invalid image box')
    p = k@np.column_stack([r, t])
    terms = [p[0]-b[2]*p[2], b[0]*p[2]-p[0], p[1]-b[3]*p[2], b[1]*p[2]-p[1]]
    a = np.array([v[:3] for v in terms]+[-p[2, :3]])
    rhs = np.r_[[-v[3] for v in terms], p[2, 3]-.001]
    return a, rhs


def feasible(rows, cameras, bounds=None):
    parts = [region_inequalities(r['box'], cameras[r['camera']]) for r in rows]
    a, b = np.vstack([x[0] for x in parts]), np.concatenate([x[1] for x in parts])
    result = linprog(np.zeros(3), A_ub=a, b_ub=b, bounds=bounds or [(None, None)]*3, method='highs')
    if result.success:
        violation = float(np.max(a@result.x-b))
        if violation > 1e-7:
            raise RuntimeError('LP feasibility residual too large')
        return {'feasible': True, 'conditional_example_point_m': result.x.tolist(), 'max_inequality_violation': violation,
                'exact_joint_reconstructed': False}
    if result.status != 2:
        raise RuntimeError('LP failed without infeasibility certificate: '+result.message)
    return {'feasible': False, 'exact_joint_reconstructed': False}


def minimum_padding(rows, cameras, maximum=20., tolerance=.001):
    def expanded(amount):
        return [dict(r, box=(np.array(r['box'])+np.array([-amount, -amount, amount, amount])).tolist()) for r in rows]
    if feasible(rows, cameras)['feasible']:
        return 0.
    if not feasible(expanded(maximum), cameras)['feasible']:
        return None
    low, high = 0., maximum
    while high-low > tolerance:
        mid = (high+low)/2
        if feasible(expanded(mid), cameras)['feasible']:
            high = mid
        else:
            low = mid
    return high


def run():
    source = Source()
    old = json.loads((ROOT/'long_hand_regions.json').read_text(encoding='utf-8'))['observations']
    new = json.loads((ROOT/'counterview_regions.json').read_text(encoding='utf-8'))['observations']
    primary = json.loads((ROOT/'long_hand_results.json').read_text(encoding='utf-8'))
    result = {'source_sha256': {str(ROOT/'counterview_regions.json'): sha(ROOT/'counterview_regions.json'),
              str(ROOT/'COUNTERVIEW_REGION_PROTOCOL.md'): sha(ROOT/'COUNTERVIEW_REGION_PROTOCOL.md')}, 'segments': {}}
    for segment in ['l_hand', 'l_larm']:
        all_rows = [r for r in old+new if r['frame'] == 950 and r['segment'] == segment]
        center = source.origins[segment](951)
        bounds = list(zip(center-.2, center+.2))
        report = {}
        for name, ids in [('views20_21', [20, 21]), ('views17_20_21', [17, 20, 21])]:
            rows = [r for r in all_rows if r['camera'] in ids]
            report[name] = {'unbounded': feasible(rows, source.cameras), 'within200mm': feasible(rows, source.cameras, bounds)}
            report[name]['minimum_shared_box_padding_px'] = minimum_padding(rows, source.cameras)
            report[name]['padding_is_accuracy_or_confidence_bound'] = False
        report['new_view_model_residuals'] = []
        for row in [r for r in new if r['segment'] == segment]:
            item = dict(row)
            for name in ['M0', 'M2']:
                m = primary['models'][name]
                uv = project(source.point(row, m['offset_frames'], m['translation_m']), source.cameras[row['camera']])[0]
                item[name+'_outside_px'] = float(np.linalg.norm(box_residual(uv, row['box'])))
                item[name+'_pixel'] = uv.tolist()
            report['new_view_model_residuals'].append(item)
        result['segments'][segment] = report
    result['calibration_updated'] = False; result['anatomical_accuracy_established'] = False
    path = ROOT/'counterview_results.json'; path.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result))
    return result


if __name__ == '__main__':
    run()
