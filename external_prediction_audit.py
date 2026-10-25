"""Rescore frozen author predictions; no model training or target calibration."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import loadmat


def r2(y,p):
    den=np.sum((y-np.mean(y))**2)
    return float(1-np.sum((y-p)**2)/den) if den>0 else None


def score_arrays(truth,pred):
    truth,pred=np.asarray(truth,dtype=float),np.asarray(pred,dtype=float)
    if truth.ndim!=2 or truth.shape!=pred.shape or min(truth.shape)<2:
        raise ValueError('Matching athlete by repeated-trial matrices required')
    if not np.isfinite(truth).all() or not np.isfinite(pred).all():raise ValueError('Nonfinite values')
    i,j=np.triu_indices(truth.shape[1],1)
    delta=truth[:,j]-truth[:,i];pdelta=pred[:,j]-pred[:,i]
    baseline=np.abs(delta).mean(axis=1);error=np.abs(pdelta-delta).mean(axis=1)
    diff=error-baseline
    rng=np.random.default_rng(20261025)
    boot=diff[rng.integers(0,len(diff),(10000,len(diff)))].mean(axis=1)
    def subset(mask):
        return {'pairs':int(mask.sum()),'direction_accuracy':float(np.mean(np.sign(pdelta[mask])==np.sign(delta[mask]))) if mask.any() else None,
                'delta_mae_kmh':float(np.mean(np.abs(pdelta[mask]-delta[mask]))) if mask.any() else None,
                'zero_delta_mae_kmh':float(np.mean(np.abs(delta[mask]))) if mask.any() else None}
    return {'absolute_mae_kmh':float(np.abs(pred-truth).mean()),'pitch_r2':r2(truth,pred),
            'athlete_mean_r2':r2(truth.mean(axis=1),pred.mean(axis=1)),
            'delta_mae_kmh':float(error.mean()),'zero_delta_mae_kmh':float(baseline.mean()),
            'mae_difference_kmh':float(diff.mean()),'difference_bootstrap95':np.quantile(boot,[.025,.975]).tolist(),
            'athletes_better_than_zero':int(np.sum(diff< -1e-10)),
            'observed_tie_pairs':int(np.sum(np.abs(delta)<=1e-6)),
            'non_tie':subset(np.abs(delta)>1e-6),'at_least_1kmh':subset(np.abs(delta)>=1),
            'per_athlete_mae':error.tolist(),'per_athlete_zero_mae':baseline.tolist()}


def check_labels(csv_values,mat_values):
    if np.shape(csv_values)!=np.shape(mat_values) or not np.allclose(csv_values,mat_values,atol=1e-4,rtol=0):
        raise ValueError('CSV-MAT labels disagree; do not reorder by speed')


def run(root=Path('artifacts/external/takamido')):
    root=Path(root);data=root/'ball_speed_pred_colab/data';folders=sorted(data.iterdir())
    if [p.name for p in folders]!=[f'Sub{i:03d}' for i in range(1,51)]:raise ValueError('Unexpected subjects')
    truths=[];sources=[]
    for folder in folders:
        paths=sorted(folder.glob('*.mat'))
        if len(paths)!=5:raise ValueError('Expected five trials')
        y=[]
        for path in paths:
            d=loadmat(path)
            if d['X'].shape!=(101,15,3) or d['Y'].size!=1 or not np.isfinite(d['X']).all():raise ValueError('Unexpected data')
            y.append(float(d['Y'].item()));sources.append({'path':str(path.relative_to(root)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        truths.append(y)
    truth=np.array(truths);models=[];prediction_arrays={}
    dirs=sorted((root/'MATLAB_code/prediction/base_models').iterdir())
    if len(dirs)!=12:raise ValueError('Expected all twelve models')
    examples=[]
    for directory in dirs:
        predictions=[]
        if len(list(directory.glob('Sub*.csv')))!=50:raise ValueError('Missing predictions')
        for k,folder in enumerate(folders):
            path=directory/(folder.name+'.csv');d=pd.read_csv(path)
            if d.shape!=(5,2) or list(d.columns)!=['Value1','Value2']:raise ValueError('Unexpected prediction shape')
            check_labels(d.iloc[:,1].to_numpy(),truth[k]);predictions.append(d.iloc[:,0].to_numpy())
            sources.append({'path':str(path.relative_to(root)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        pred=np.array(predictions)
        prediction_arrays[directory.name]=pred
        models.append({'model':directory.name,**score_arrays(truth*1.609344,pred*1.609344)})
        if directory.name=='TF_4_64_128':
            examples=[{'subject':folders[k].name,'observed_kmh':(truth[k]*1.609344).tolist(),'predicted_kmh':(pred[k]*1.609344).tolist()} for k in range(3)]
    names=sorted(prediction_arrays)
    duplicates=[[left,right] for i,left in enumerate(names) for right in names[i+1:]
                if np.array_equal(prediction_arrays[left],prediction_arrays[right])]
    report={'subjects':50,'trials':250,'pairs':500,'models':models,'examples_first_three':examples,
            'exact_duplicate_prediction_pairs':duplicates,
            'subjects_monotone_nonincreasing_speed':int(np.all(np.diff(truth,axis=1)<=1e-4,axis=1).sum()),
            'subjects_monotone_nondecreasing_speed':int(np.all(np.diff(truth,axis=1)>=-1e-4,axis=1).sum()),
            'sources':sources,'scope':'Frozen author prediction re-scoring; 5 fastest pitches per subject. All 10 unordered pairs, not chronological changes, interventions, independent pairs or a Real2Sim comparison.',
            'input_observation':'X:101x15x3 body landmarks; Y:scalar speed. No directly tracked ball trajectory or hand/finger cluster in these MAT arrays.'}
    out=Path('artifacts/external-benchmark');out.mkdir(parents=True,exist_ok=True)
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    for m in models:
        print(m['model'],{k:round(m[k],4) for k in ('pitch_r2','athlete_mean_r2','absolute_mae_kmh','delta_mae_kmh','zero_delta_mae_kmh')},m['difference_bootstrap95'])
    print('monotone',report['subjects_monotone_nonincreasing_speed'],report['subjects_monotone_nondecreasing_speed'])
    return report


if __name__=='__main__':run()
