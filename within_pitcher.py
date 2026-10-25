"""Do held-out predictions track within-pitcher observed speed differences?"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def evaluate_changes(heldout):
    methods=['predicted','scalar_linear','zero']
    rows=[]
    for (mode,user),g in heldout.groupby(['mode','user']):
        g=g.sort_values('session_pitch')
        if len(g)<2:continue
        ref=g.iloc[0]
        for row in g.iloc[1:].itertuples():
            actual=float(row.observed_kmh-ref.observed_kmh)
            for method in methods+['no_change']:
                pred=0. if method=='no_change' else float(getattr(row,method+'_kmh')-ref[method+'_kmh'])
                rows.append({'mode':mode,'user':user,'reference_pitch':ref.session_pitch,
                             'evaluated_pitch':row.session_pitch,'method':method,
                             'observed_delta_kmh':actual,'predicted_delta_kmh':pred,
                             'absolute_delta_error_kmh':abs(pred-actual)})
    frame=pd.DataFrame(rows)
    if frame.empty:raise ValueError('No repeated pitches')
    summary=[]
    for (mode,method),g in frame.groupby(['mode','method']):
        mean=g.groupby('user').absolute_delta_error_kmh.mean().mean()
        summary.append({'mode':mode,'method':method,'athletes':int(g.user.nunique()),
                        'evaluation_pitches':len(g),'athlete_balanced_delta_mae_kmh':float(mean)})
    return frame,summary


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',type=Path,default=Path('artifacts/release-calibration/heldout.csv'))
    parser.add_argument('--output',type=Path,default=Path('artifacts/within-pitcher'))
    args=parser.parse_args()
    frame,summary=evaluate_changes(pd.read_csv(args.input))
    args.output.mkdir(parents=True,exist_ok=True)
    frame.to_csv(args.output/'changes.csv',index=False)
    report={'reference':'First lexicographic pitch per athlete, not asserted chronological',
            'scope':'Same-session observed movement/speed variation; no intervention and no causal effect claim.',
            'fitting':'Uses existing athlete-disjoint predictions; no model refit on evaluation athlete.',
            'summary':summary,'intervention_validated':False}
    (args.output/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
