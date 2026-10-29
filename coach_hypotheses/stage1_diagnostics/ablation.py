"""Post-primary matched-penalty diagnostic, never changes selector or gate."""
import numpy as np
import pandas as pd

from ..stage1.modeling import demean, fit_ridge, predict_ridge, prediction_summary
from .diagnostics import FAMILIES


def matched_penalty(frame):
    ordered=frame.sort_values(['user','session_pitch'])
    groups=ordered.user.to_numpy(str)
    y=demean(ordered.pitch_speed_kmh.to_numpy(float),groups)
    records=[]
    for family,columns in FAMILIES.items():
        if family=='mae14':
            continue
        x=demean(ordered[columns].to_numpy(float),groups)
        for alpha in (100.,1000.):
            rows=[]
            for held in np.unique(groups):
                test=groups==held
                model=fit_ridge(x[~test],y[~test],groups[~test],alpha)
                prediction=demean(predict_ridge(model,x[test]),groups[test])
                rows.extend({'user':held,'truth':float(t),'prediction':float(p),'baseline':0.}
                            for t,p in zip(y[test],prediction))
            summary,_=prediction_summary(pd.DataFrame(rows))
            records.append({'family':family,'fixed_alpha':alpha,
                            'mae_kmh':summary['player_balanced_model_mae_kmh'],
                            'difference_from_null_kmh':summary['paired_mae_difference_kmh']})
    return pd.DataFrame(records)
