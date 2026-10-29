import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from ..stage1.metrics import COLUMNS
from ..stage1.modeling import predict_ridge
from ..stage1.test_metrics import fixture_tables, fixture_metadata, fixture_poi
from ..stage1.test_modeling import fixture_features
from ..stage1.run import file_hash
from .diagnostics import (WAVE, EXTENDED, FAMILIES, CHANNELS, verify_sources, phase_features,
                          add_waveforms, data_diagnostics, fit_model, balanced_mae,
                          inner_choices, compare_models, permutation_controls)
from .run import main, analyze, synthetic_control


def diagnostic_fixture():
    frame = fixture_features()
    for i, column in enumerate(WAVE):
        frame[column] = frame[EXTENDED[i % len(EXTENDED)]]
    return frame


class DiagnosticTests(unittest.TestCase):
    def test_verified_sources_detect_changed_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'source.csv').write_bytes(b'original')
            (p/'manifest.json').write_text(json.dumps({'sources':{'source.csv':{'sha256':file_hash(p/'source.csv')}}}),encoding='utf-8')
            self.assertTrue(verify_sources(p,p)['source.csv'])
            (p/'source.csv').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'changed_frozen'):
                verify_sources(p,p)

    def test_phase_features_known_linear_trajectory(self):
        time=np.linspace(0,2,21);signals={c:time*10 for c in CHANNELS}
        result=phase_features(time,signals,.5,1.5)
        self.assertEqual([result[f'phase_{CHANNELS[0]}_{i}'] for i in range(5)],[5,7.5,10,12.5,15])
        with self.assertRaises(ValueError):phase_features(time,signals,1.5,.5)

    def test_add_waveforms_and_data_quality_fixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'full_sig').mkdir()
            source=fixture_tables()
            for name,f in source.items():
                f['session_pitch']='10_1';f.to_csv(p/f'full_sig/{name}.csv',index=False)
            frame=diagnostic_fixture().iloc[:1].drop(columns=WAVE).copy()
            frame['session_pitch']='10_1'
            all_conditions=pd.concat([frame.assign(cutoff_hz=c) for c in (10,20,30)])
            result=add_waveforms(p,all_conditions)
            self.assertEqual(result.shape[0],3)
            self.assertTrue(np.isfinite(result[WAVE]).all().all())
        frame=diagnostic_fixture()
        result=pd.concat([frame.assign(cutoff_hz=c) for c in (10,20,30)])
        report,features=data_diagnostics(result)
        self.assertEqual((report['players'],report['pitches']),(6,18))
        self.assertEqual(len(features),14)
        self.assertTrue((features.median_filter_range==0).all())

    def test_loss_control_and_failed_optimizer_not_hidden(self):
        x=np.arange(20,dtype=float)[:,None];y=2*x[:,0];y[-1]+=200
        groups=np.array([str(i) for i in range(20)])
        ridge=fit_model(x,y,groups,.01)
        mae=fit_model(x,y,groups,.01,'mae')
        self.assertLess(np.abs(predict_ridge(mae,x)-2*x[:,0]).mean(),np.abs(predict_ridge(ridge,x)-2*x[:,0]).mean())
        with self.assertRaisesRegex(ValueError,'invalid_loss'):fit_model(x,y,groups,1,'other')
        with patch('coach_hypotheses.stage1_diagnostics.diagnostics.minimize') as optimizer:
            optimizer.return_value.success=False;optimizer.return_value.message='example'
            with self.assertRaisesRegex(RuntimeError,'optimizer_failed'):fit_model(x,y,groups,1,'mae')

    def test_inner_null_tie_and_weighting(self):
        groups=np.array(['a','a','b','b','c','c'])
        choices,best,scores=inner_choices({'original12':np.ones((6,2))},np.zeros(6),groups)
        self.assertEqual(best['family'],'null')
        self.assertEqual(choices['original12']['alpha'],100)
        self.assertEqual(balanced_mae(np.array([0.,0.,9.]),np.zeros(3),np.array(['a','a','b'])),4.5)

    def test_model_selection_has_no_held_label_leak(self):
        frame=diagnostic_fixture()
        before,folds,_=compare_models(frame)
        frame.loc[frame.user=='2','pitch_speed_kmh']+=[10,-20,40]
        after,_,_=compare_models(frame)
        a=before[before.user=='2'].sort_values(['family','session_pitch'])
        b=after[after.user=='2'].sort_values(['family','session_pitch'])
        np.testing.assert_allclose(a.prediction,b.prediction)
        self.assertEqual(a.selected_family.tolist(),b.selected_family.tolist())
        self.assertFalse(folds.held_player_in_training.any())

    def test_permutation_control_includes_actual_and_train_shuffles(self):
        result=permutation_controls(diagnostic_fixture(),repetitions=2)
        self.assertEqual(result.training_labels_shuffled.tolist(),[False,True,True])
        self.assertTrue(np.isfinite(result.mae_kmh).all())

    def test_synthetic_known_signal_is_recovered(self):
        self.assertTrue(synthetic_control()['recovers_known_signal'])

    def test_cli_dispatch_and_output_guard(self):
        with patch('sys.argv',['diagnostics','--data','example']),patch('coach_hypotheses.stage1_diagnostics.run.analyze',return_value={'selector_gate':{},'selector_counts':{}}) as mocked,patch('builtins.print'):
            main();mocked.assert_called_once()
        with patch('sys.argv',['diagnostics','--data','example','--output',str(Path.home())]):
            with self.assertRaises(SystemExit):main()

    def test_analyze_orchestration_with_verified_pipeline_contract(self):
        frame=diagnostic_fixture();frame=pd.concat([frame.assign(cutoff_hz=c) for c in (10,20,30)])
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)
            # Real algorithm tests above are separate; this checks output orchestration.
            fake_predictions,folds,scores=compare_models(frame[frame.cutoff_hz==20])
            previous=fake_predictions[fake_predictions.family=='original12'].drop(columns=['family','selected_family'])
            def read(path,*args,**kwargs):
                return frame.copy() if str(path).endswith('pitch_metrics.csv') else previous.copy()
            with patch('coach_hypotheses.stage1_diagnostics.run.verify_sources',return_value={'example':True}),patch('coach_hypotheses.stage1_diagnostics.run.add_waveforms',side_effect=lambda data,f:f),patch('coach_hypotheses.stage1_diagnostics.run.pd.read_csv',side_effect=read),patch('coach_hypotheses.stage1_diagnostics.run.compare_models',return_value=(fake_predictions,folds,scores)),patch('coach_hypotheses.stage1_diagnostics.run.synthetic_control',return_value={'synthetic_only':True}):
                result=analyze(path,path/'output')
            self.assertEqual(result['selector_gate']['verdict'],'FAIL')
            self.assertTrue((path/'output/summary.json').is_file())


if __name__=='__main__':unittest.main()
