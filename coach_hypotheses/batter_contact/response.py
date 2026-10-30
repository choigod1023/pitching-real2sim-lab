"""Restore frozen MLB models; do not relabel statistical miss as physical contact."""
import json
from pathlib import Path
import numpy as np
from ..temporal_quality import model as m


def restore(path, expected_sha256):
    if m.digest(path)!=expected_sha256:raise ValueError('fit_provenance_mismatch')
    audit=json.loads(Path(path).read_text(encoding='utf-8'))
    fitted=m.Logistic();fitted.features=m.Features()
    fitted.features.model=audit['model'];fitted.features.quality=audit['model'].endswith('1')
    fitted.features.splines={}
    for name,knots in audit['spline_knots'].items():
        s=m.Spline();s.knots=np.asarray(knots,float);s.lo=float(s.knots[0]);s.hi=float(s.knots[-1])
        fitted.features.splines[name]=s
    e=m.old.Encoder();e.groups=audit['group_column_order'];e.indices={g:i for i,g in enumerate(e.groups)}
    e.n_cont=audit['continuous_columns'];e.scale=np.asarray(audit['training_scale'],float)
    e.means={g:np.asarray(v,float) for g,v in audit['group_means'].items()}
    e.fallback=np.asarray(audit['unknown_group_mean'],float);e.width=len(audit['feature_names'])
    fitted.encoder=e;fitted.batters={int(k):v for k,v in audit['batter_columns'].items()}
    fitted.beta=np.asarray(audit['coefficients'],float)
    if (len(fitted.beta)!=1+len(e.groups)+e.width+len(fitted.batters)
        or not np.isfinite(fitted.beta).all() or np.any(e.scale<=0)):
        raise ValueError('invalid_saved_design')
    return fitted


def joint_branches(swing, labelled_miss_given_swing):
    s,p=np.broadcast_arrays(np.asarray(swing,float),np.asarray(labelled_miss_given_swing,float))
    if not np.isfinite(s).all() or not np.isfinite(p).all() or np.any((s<0)|(s>1)|(p<0)|(p>1)):
        raise ValueError('invalid_probabilities')
    return dict(take=1-s,swing_labelled_miss=s*p,swing_other=s*(1-p))


class BatterResponse:
    def __init__(self, summary_path, fit_root, model='GB1'):
        if model not in ('GB0','GB1'):raise ValueError('unsupported_response_model')
        summary=json.loads(Path(summary_path).read_text(encoding='utf-8'));self.models={}
        for endpoint in ('swing',)+m.TARGETS:
            meta=next(f for f in summary['analyses'][endpoint]['fits'] if f['model']==model)
            if not meta['success']:raise ValueError('nonconverged_response_model')
            filename=f'{endpoint}_{model}_10.json'
            self.models[endpoint]=restore(Path(fit_root)/endpoint/filename,meta['fit_audit_sha256'])

    def predict(self, rows, *, data_source):
        if data_source!='mlb_observed':raise ValueError('no_validated_body_to_mlb_feature_mapping')
        if not rows:raise ValueError('empty_observation_list')
        for r in rows:
            count=r.get('pre_count') or {}
            if (any(type(r.get(k)) is not int for k in ('gamePk','atBatIndex','pitchIndex','pitcherId','batterId'))
                or r.get('pitchType')!='FF' or r.get('batSide') not in ('L','R') or r.get('pitchHand') not in ('L','R')
                or type(count.get('balls')) is not int or not 0<=count['balls']<=3
                or type(count.get('strikes')) is not int or not 0<=count['strikes']<=2
                or any(type(r.get(k)) not in (int,float) or not np.isfinite(r[k]) for k in m.old.FIELDS)):
                raise ValueError('invalid_observed_pitch_contract')
        probabilities={k:f.predict(rows)[0] for k,f in self.models.items()}
        result=[]
        for i,r in enumerate(rows):
            output=dict(key=[r['gamePk'],r['atBatIndex'],r['pitchIndex']],batterId=r['batterId'],
                        swing_probability=float(probabilities['swing'][i]),physical_contact_probability=None,
                        actual_swing_trajectory_predicted=False,causal_body_adjustment_effect=False)
            for endpoint in m.TARGETS:
                p=probabilities[endpoint][i]
                output[endpoint]=dict(labelled_miss_given_swing=float(p),
                    branches={k:float(v) for k,v in joint_branches(probabilities['swing'][i],p).items()})
            result.append(output)
        return result
