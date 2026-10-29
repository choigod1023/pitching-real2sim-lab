from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from ..stage1.modeling import demean,fit_ridge,predict_ridge
from ..stage1_diagnostics.diagnostics import WAVE
from .benchmark import fit_mapping,map_features,prediction_grid,select_models,nested_predictions,run,main


def fixture():
    rows=[]
    for player in range(6):
        for trial,delta in enumerate([-1.,0.,1.]):
            row={name:player*4+delta*(i+1) for i,name in enumerate(WAVE)}
            rows.append(dict(row,user=str(player),session_pitch=f'{player}_{trial}',pitch_speed_kmh=100+player*2+delta))
    return pd.DataFrame(rows)


class ReplacementTests(unittest.TestCase):
    def test_mapping_is_train_only_and_fixed_random_features(self):
        x=np.array([[1.,2.],[3.,2.],[5.,2.],[7.,2.]])
        groups=np.array(['a','a','b','b'])
        first=fit_mapping(x,groups,'rff_1.0');second=fit_mapping(x,groups,'rff_1.0')
        np.testing.assert_array_equal(first[3],second[3])
        scale=first[1].copy()
        a=map_features(first,np.array([[10.,2.],[20.,2.]]),np.array(['c','c']))
        b=map_features(first,np.array([[10.,999.],[20.,-999.]]),np.array(['c','c']))
        np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(first[1],scale)
        self.assertEqual(a.shape,(2,64));np.testing.assert_allclose(a.mean(axis=0),0,atol=1e-14)
        with self.assertRaises(ValueError):fit_mapping(x,groups,'other')

    def test_linear_grid_matches_existing_weighted_ridge(self):
        frame=fixture();groups=frame.user.to_numpy(str)
        x=demean(frame[WAVE].to_numpy(float),groups);y=demean(frame.pitch_speed_kmh.to_numpy(float),groups)
        test=groups=='2'
        grid=prediction_grid(x[~test],y[~test],groups[~test],x[test],groups[test],'linear')
        for alpha,pred in grid.items():
            model=fit_ridge(x[~test],y[~test],groups[~test],alpha)
            expected=demean(predict_ridge(model,x[test]),groups[test])
            np.testing.assert_allclose(pred,expected,atol=1e-10)

    def test_nonlinear_control_recovers_known_quadratic_signal(self):
        x=np.zeros((54,20));x[:,0]=np.tile(np.linspace(-2,2,9),6)
        groups=np.repeat(np.arange(6).astype(str),9);y=demean(x[:,0]**2,groups);test=groups=='0'
        linear=prediction_grid(x[~test],y[~test],groups[~test],x[test],groups[test],'linear')[.01]
        nonlinear=prediction_grid(x[~test],y[~test],groups[~test],x[test],groups[test],'rff_2.0')[.01]
        self.assertLess(np.mean(abs(nonlinear-y[test])),.5*np.mean(abs(linear-y[test])))

    def test_null_selection_when_no_target_signal(self):
        frame=fixture();groups=frame.user.to_numpy(str);x=demean(frame[WAVE].to_numpy(float),groups)
        best,_,_,_=select_models(x,np.zeros(len(x)),groups)
        self.assertEqual(best['family'],'null')
        with self.assertRaisesRegex(ValueError,'too_few'):select_models(x,np.zeros(len(x)),np.array(['a']*len(x)))

    def test_held_player_labels_cannot_change_predictions_or_selection(self):
        frame=fixture();before,audit,_=nested_predictions(frame)
        frame.loc[frame.user=='2','pitch_speed_kmh']+=[40.,-100.,30.]
        after,_,_=nested_predictions(frame)
        a=before[before.user=='2'].sort_values(['model','session_pitch']);b=after[after.user=='2'].sort_values(['model','session_pitch'])
        np.testing.assert_allclose(a.prediction,b.prediction);self.assertEqual(a.selected_family.tolist(),b.selected_family.tolist())
        self.assertFalse(audit.overlap.any())

    def test_run_orchestration_and_output_guard(self):
        frame=fixture();all_rows=pd.concat([frame.assign(cutoff_hz=c) for c in (10,20,30)])
        predicted,audit,scores=nested_predictions(frame)
        old=predicted[predicted.model=='linear'].assign(family='waveform20')
        previous=pd.concat([old.assign(cutoff_hz=c) for c in (10,20,30)])
        def read(path,*args,**kwargs):
            return previous.copy() if str(path).endswith('predictions.csv') else all_rows.copy()
        with tempfile.TemporaryDirectory() as tmp,patch('coach_hypotheses.replacement_review.benchmark.pd.read_csv',side_effect=read),patch('coach_hypotheses.replacement_review.benchmark.nested_predictions',return_value=(predicted,audit,scores)):
            result=run(Path(tmp));self.assertEqual(result['gate']['verdict'],'FAIL');self.assertTrue(all(result['linear_reproduced'].values()))
        with patch('sys.argv',['benchmark']),patch('coach_hypotheses.replacement_review.benchmark.run',return_value={'gate':{},'selector_counts':{}}) as mocked,patch('builtins.print'):
            main();mocked.assert_called_once()
        with patch('sys.argv',['benchmark','--output',str(Path.home())]):
            with self.assertRaises(SystemExit):main()


if __name__=='__main__':unittest.main()
