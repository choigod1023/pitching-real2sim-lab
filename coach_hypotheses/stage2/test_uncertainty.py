import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from .uncertainty import (FACTOR_NAMES, file_hash, prepare_case, velocities, speeds,
                          rigid_transform_velocity, grouped_mc, convergence, recorded_comparison)
from .run import main, run


def synthetic_case():
    rotation = np.broadcast_to(np.eye(3), (3,3,3,3)).copy()
    derivative = np.zeros_like(rotation)
    derivative[:, :, 1, 0] = 100.
    derivative[:, :, 0, 1] = -100.
    return {'rate_hz': 360., 'wrist_velocity': np.zeros((3,3)),
            'rotation': rotation, 'rotation_derivative': derivative}


def synthetic_data(path):
    (path/'c3d').mkdir()
    (path/'full_sig').mkdir()
    (path/'c3d/example.c3d').write_bytes(b'synthetic-not-real-c3d')
    pd.DataFrame({'session_pitch': ['3034_1'], 'filename_new': ['example.c3d']}).to_csv(path/'metadata.csv', index=False)
    t = np.arange(721)/360.
    frame = pd.DataFrame({'session_pitch': '3034_1', 'time': np.round(t, 4), 'fp_10_time': .8, 'BR_time': 1.2})
    shoulder = np.tile([0., 0., 1.5], (len(t),1))
    elbow = shoulder + np.column_stack([.3*np.cos(t), np.zeros(len(t)), .3*np.sin(t)])
    wrist = elbow + np.column_stack([.25*np.cos(2*t), np.zeros(len(t)), .25*np.sin(2*t)])
    for name, value in zip(('shoulder','elbow','wrist'),(shoulder,elbow,wrist)):
        for i, axis in enumerate('xyz'):
            frame[name+'_jc_'+axis] = value[:, i]
    frame.to_csv(path/'full_sig/landmarks.csv', index=False)
    fin = wrist + np.column_stack([.1*np.cos(3*t), np.zeros(len(t)), .1*np.sin(3*t)])
    raw = np.ones((4,3,len(t)))
    for i, values in enumerate((fin, wrist+[0,.02,0], wrist-[0,.02,0])):
        raw[:3,i,:] = values.T
    return {'parameters': {'POINT': {'LABELS': {'value':['RFIN','RWRA','RWRB']},
                                     'RATE': {'value':[360.]}, 'UNITS': {'value':['m']}}},
            'data': {'points': raw}}


class ScenarioTests(unittest.TestCase):
    def test_kinematic_distance_and_distance_rate_units(self):
        case = synthetic_case()
        u = np.full((1,7), .5)
        np.testing.assert_allclose(velocities(case,u),[[0.,15.,0.]])
        self.assertAlmostEqual(speeds(case,u)[0],54.)
        u[:, 1] = 1.
        np.testing.assert_allclose(velocities(case,u),[[.5,15.,0.]])
        np.testing.assert_allclose(velocities(case,u,relative_motion=False),[[0.,15.,0.]])
        u[:, 0] = 1.
        self.assertGreater(speeds(case,u)[0],54.)

    def test_local_extremes_and_offset_reach(self):
        u = np.array([[0,0,0,0,1,.5,0],[1,1,1,1,1,.5,0]],float)
        self.assertTrue(np.isfinite(speeds(synthetic_case(),u,.017)).all())
        u[:,0] = -1
        with self.assertRaisesRegex(ValueError, 'unit_samples'):
            velocities(synthetic_case(),u)
        with self.assertRaises(ValueError):
            speeds(synthetic_case(),np.full((1,7),.5),-.1)

    def test_common_translation_and_rotation_preserve_speed(self):
        v=np.array([[1.,2.,3.],[4.,5.,6.]])
        r=Rotation.from_rotvec([.2,.3,-.4]).as_matrix()
        output=rigid_transform_velocity(v,r,np.array([.017,.017,.017]))
        np.testing.assert_allclose(np.linalg.norm(v,axis=1),np.linalg.norm(output,axis=1),atol=1e-12)
        with self.assertRaises(ValueError):
            rigid_transform_velocity(v,r*2,np.zeros(3))

    def test_additive_group_indices_and_reproducibility(self):
        model=lambda x: x[:,0]+2*x[:,2]
        a=grouped_mc(model,32768,123)
        self.assertAlmostEqual(a['first_order'][FACTOR_NAMES[0]],.2,delta=.02)
        self.assertAlmostEqual(a['first_order'][FACTOR_NAMES[1]],.8,delta=.02)
        self.assertAlmostEqual(a['total_order'][FACTOR_NAMES[0]],.2,delta=.02)
        self.assertEqual(a['first_order'][FACTOR_NAMES[3]],0.)
        self.assertEqual(a,grouped_mc(model,32768,123))

    def test_product_interaction_analytic(self):
        result=grouped_mc(lambda x:x[:,0]*x[:,2],65536,84)
        self.assertAlmostEqual(result['first_order'][FACTOR_NAMES[0]],3/7,delta=.025)
        self.assertAlmostEqual(result['first_order'][FACTOR_NAMES[1]],3/7,delta=.025)
        self.assertAlmostEqual(result['total_order'][FACTOR_NAMES[0]],4/7,delta=.025)
        self.assertAlmostEqual(result['interaction_residual'],1/7,delta=.03)

    def test_degenerate_design_and_output_fail(self):
        with self.assertRaises(ValueError):
            grouped_mc(lambda x:np.zeros(len(x)),32,1)
        with self.assertRaises(ValueError):
            grouped_mc(lambda x:x[:,0],2,1)
        with self.assertRaises(ValueError):
            grouped_mc(lambda x:np.full(len(x),np.nan),32,1)

    def test_convergence_pass_and_explicit_fail(self):
        first=grouped_mc(lambda x:x[:,0],8192,1)
        second=dict(first,seed=2)
        self.assertEqual(convergence([first,second])['verdict'],'PASS')
        second['mean_kmh']+=.2
        self.assertEqual(convergence([first,second])['verdict'],'FAIL')

    def test_loader_without_radar_and_legacy_regression(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fake=synthetic_data(root)
            with patch('coach_hypotheses.stage2.uncertainty.ezc3d.c3d',return_value=fake):
                case=prepare_case(root,'3034_1')
            self.assertLess(case['legacy_regression_error_kmh'],1e-9)
            self.assertEqual(case['rate_hz'],360.)
            self.assertEqual(len(case['sources']),3)
            # Metadata deliberately lacks pitch_speed_mph.
            self.assertNotIn('recorded_speed_kmh',case)
            fake['data']['points'][3,0,10]=-1
            with patch('coach_hypotheses.stage2.uncertainty.ezc3d.c3d',return_value=fake):
                with self.assertRaisesRegex(ValueError,'invalid_marker'):
                    prepare_case(root,'3034_1')

    def test_hash_and_separate_recorded_comparison(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'example').write_bytes(b'abc')
            self.assertEqual(file_hash(root/'example'),'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad')
            pd.DataFrame({'session_pitch':['3034_1'],'pitch_speed_mph':[80.]}).to_csv(root/'metadata.csv',index=False)
            output=recorded_comparison(root,[{'pitch':'3034_1','nominal_15cm_kmh':120.,'primary':{'mean_kmh':121.}}])
            self.assertAlmostEqual(output[0]['recorded_speed_kmh'],128.74752)
            self.assertAlmostEqual(output[0]['nominal_difference_kmh'],-8.74752)

    def test_cli_dispatch_and_output_boundary(self):
        with patch('sys.argv',['stage2','--data','example']),patch('coach_hypotheses.stage2.run.run',return_value={'summaries':[], 'convergence':[]}) as mocked,patch('builtins.print'):
            main();mocked.assert_called_once()
        with patch('sys.argv',['stage2','--data','example','--output',str(Path.home())]):
            with self.assertRaises(SystemExit):main()

    def test_run_freezes_outputs_before_recorded_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)
            case=dict(synthetic_case(),pitch='3034_1',sources={},legacy_regression_error_kmh=0.,lengths_m=[.3,.25],br_s=1.)
            fake=grouped_mc(lambda u:u[:,0],32,1)
            def compare(data,rows):
                self.assertTrue((output/'scenario_results.json').exists())
                self.assertTrue((output/'mc_designs.json').exists())
                return []
            with patch('coach_hypotheses.stage2.run.PITCHES',('3034_1',)),patch('coach_hypotheses.stage2.run.prepare_case',return_value=case),patch('coach_hypotheses.stage2.run.grouped_mc',return_value=fake),patch('coach_hypotheses.stage2.run.recorded_comparison',side_effect=compare):
                result=run(Path('unused'),output)
            self.assertEqual(len(result['summaries']),1)
            self.assertIsInstance(json.loads((output/'manifest.json').read_text()),dict)


if __name__=='__main__':unittest.main()
