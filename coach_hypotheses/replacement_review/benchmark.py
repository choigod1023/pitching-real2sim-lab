"""Same-input nonlinear control; not a counterfactual biomechanics simulator."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..stage1.modeling import demean, player_weights, prediction_summary, scientific_gate
from ..stage1_diagnostics.diagnostics import WAVE

ALPHAS=(.01,.1,1.,10.,100.,1000.)
FAMILIES=('linear','rff_0.5','rff_1.0','rff_2.0')


def fit_mapping(x, groups, family):
    weights=player_weights(groups)
    center=np.average(x,axis=0,weights=weights)
    scale=np.sqrt(np.average((x-center)**2,axis=0,weights=weights))
    constant=scale<1e-10;scale[constant]=1.
    if family=='linear':
        frequency=phase=None
    elif family in FAMILIES:
        rng=np.random.default_rng(20261029)
        frequency=rng.normal(size=(x.shape[1],64))*float(family.split('_')[1])/np.sqrt(x.shape[1])
        phase=rng.uniform(0,2*np.pi,64)
    else:
        raise ValueError('unknown_feature_family')
    return center,scale,constant,frequency,phase


def map_features(mapping, x, groups):
    center,scale,constant,frequency,phase=mapping
    z=(x-center)/scale;z[:,constant]=0.
    if frequency is not None:
        z=np.sqrt(2/64)*np.cos(z@frequency+phase)
    return demean(z,groups)


def prediction_grid(x, y, groups, test_x, test_groups, family):
    mapping=fit_mapping(x,groups,family)
    z=map_features(mapping,x,groups);test=map_features(mapping,test_x,test_groups)
    weights=player_weights(groups)
    gram=z.T@(weights[:,None]*z)
    eigenvalues,eigenvectors=np.linalg.eigh(gram)
    eigenvalues=np.maximum(eigenvalues,0.)
    right=eigenvectors.T@(z.T@(weights*y))
    projected=test@eigenvectors
    return {alpha:demean(projected@(right/(eigenvalues+alpha)),test_groups) for alpha in ALPHAS}


def select_models(x, y, groups):
    players=np.unique(groups)
    if len(players)<2:
        raise ValueError('too_few_inner_players')
    folds=np.array_split(players,min(5,len(players)))
    scores={(family,alpha):[] for family in FAMILIES for alpha in ALPHAS}
    for held in folds:
        test=np.isin(groups,held)
        for family in FAMILIES:
            grid=prediction_grid(x[~test],y[~test],groups[~test],x[test],groups[test],family)
            for alpha,predicted in grid.items():
                for player in held:
                    mask=groups[test]==player
                    scores[family,alpha].append(float(np.mean(abs(predicted[mask]-y[test][mask]))))
    rows=[{'family':family,'alpha':alpha,'inner_mae':float(np.mean(errors))} for (family,alpha),errors in scores.items()]
    rows.append({'family':'null','alpha':0.,'inner_mae':float(np.average(abs(y),weights=player_weights(groups)))})
    priority={'null':0,**{family:i+1 for i,family in enumerate(FAMILIES)}}
    best=min(rows,key=lambda row:(round(row['inner_mae'],12),priority[row['family']],-row['alpha']))
    linear=min([r for r in rows if r['family']=='linear'],key=lambda r:(round(r['inner_mae'],12),-r['alpha']))
    nonlinear=min([r for r in rows if r['family'].startswith('rff')],key=lambda r:(round(r['inner_mae'],12),priority[r['family']],-r['alpha']))
    return best,linear,nonlinear,rows


def nested_predictions(frame):
    ordered=frame.sort_values(['user','session_pitch'])
    groups=ordered.user.to_numpy(str)
    x=demean(ordered[WAVE].to_numpy(float),groups)
    raw=ordered.pitch_speed_kmh.to_numpy(float);y=demean(raw,groups)
    predictions=[];audit=[];scores=[]
    for held in np.unique(groups):
        test=groups==held
        best,linear,nonlinear,inner=select_models(x[~test],y[~test],groups[~test])
        scores.extend(dict(row,held_player=held) for row in inner)
        cache={}
        for label,choice in [('selector',best),('linear',linear),('nonlinear',nonlinear)]:
            family=choice['family']
            if family=='null':
                prediction=np.zeros(test.sum())
            else:
                if family not in cache:
                    cache[family]=prediction_grid(x[~test],y[~test],groups[~test],x[test],groups[test],family)
                prediction=cache[family][choice['alpha']]
            for key,truth,value,speed in zip(ordered.session_pitch[test],y[test],prediction,raw[test]):
                predictions.append({'user':held,'session_pitch':key,'model':label,'selected_family':family,
                                    'alpha':choice['alpha'],'truth':float(truth),'prediction':float(value),
                                    'baseline':0.,'observed_speed_kmh':float(speed)})
            audit.append({'held_player':held,'model':label,**choice,'training_player_count':len(np.unique(groups[~test])),
                          'overlap':bool(held in groups[~test])})
    return pd.DataFrame(predictions),pd.DataFrame(audit),pd.DataFrame(scores)


def run(output):
    package=Path(__file__).parent;output=Path(output);output.mkdir(parents=True,exist_ok=True)
    frozen=package.parent/'stage1_diagnostics/results'
    raw=pd.read_csv(package.parent/'stage1/results/pitch_metrics.csv',dtype={'user':str,'session_pitch':str})
    wave=pd.read_csv(frozen/'waveform_features.csv',dtype={'user':str,'session_pitch':str})
    frame=wave.merge(raw[['user','session_pitch','cutoff_hz','pitch_speed_kmh']],on=['user','session_pitch','cutoff_hz'],validate='one_to_one')
    summaries={};results=[];audits=[];scores=[];reproduced={}
    for cutoff in (10,20,30):
        predicted,audit,inner=nested_predictions(frame[frame.cutoff_hz==cutoff])
        summaries[str(cutoff)]={}
        for name,block in predicted.groupby('model'):
            summary,errors=prediction_summary(block);summaries[str(cutoff)][name]=summary
            errors.to_csv(output/f'errors_{name}_{cutoff}.csv',index=False)
        previous=pd.read_csv(frozen/'predictions.csv',dtype={'user':str,'session_pitch':str})
        previous=previous[(previous.cutoff_hz==cutoff)&(previous.family=='waveform20')]
        current=predicted[predicted.model=='linear'].merge(previous,on=['user','session_pitch'],validate='one_to_one',suffixes=('_new','_old'))
        reproduced[str(cutoff)]=bool(np.allclose(current.prediction_new,current.prediction_old,rtol=0,atol=1e-8))
        if not reproduced[str(cutoff)]:raise AssertionError('linear_control_not_reproduced')
        results.append(predicted.assign(cutoff_hz=cutoff));audits.append(audit.assign(cutoff_hz=cutoff));scores.append(inner.assign(cutoff_hz=cutoff))
    result=pd.concat(results);audit=pd.concat(audits)
    result.to_csv(output/'predictions.csv',index=False);audit.to_csv(output/'fold_audit.csv',index=False)
    pd.concat(scores).to_csv(output/'inner_scores.csv',index=False)
    gate_input={str(c):{'within':summaries[str(c)]['selector']} for c in (10,20,30)}
    summary={'conditions':summaries,'gate':scientific_gate(gate_input,not audit.overlap.any() and all(reproduced.values())),
             'linear_reproduced':reproduced,'selector_counts':{str(c):result[(result.cutoff_hz==c)&(result.model=='selector')].groupby('user').selected_family.first().value_counts().to_dict() for c in (10,20,30)},
             'counterfactual_intervention_validated':False,'legacy_physical_model_replaced':False}
    (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    files=[package/'BENCHMARK_PROTOCOL.md',package.parent/'stage1/PROTOCOL.md',frozen/'waveform_features.csv',package.parent/'stage1/results/pitch_metrics.csv',Path(__file__)]
    hashes={str(path.resolve()):hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    (output/'manifest.json').write_text(json.dumps({'preregistration_commit':'93c95a8','sha256':hashes},indent=2),encoding='utf-8')
    return summary


def main():
    parser=argparse.ArgumentParser(description='Same-input nonlinear replacement feasibility')
    parser.add_argument('--output',type=Path,default=Path(__file__).parent/'artifacts/results')
    args=parser.parse_args()
    if not args.output.resolve().is_relative_to(Path(__file__).parent.resolve()):parser.error('output must stay inside replacement_review')
    summary=run(args.output);print(json.dumps({'gate':summary['gate'],'selection':summary['selector_counts']},indent=2))


if __name__=='__main__':main()
