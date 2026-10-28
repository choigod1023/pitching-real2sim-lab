import copy
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import mujoco
from same_pitch_audit import require_same_evaluation, gap_record, OUT
from baseline_calibration import endpoint
from export_model_notes import publish


class EvidenceGuardTests(unittest.TestCase):
    def test_matching_identity_can_compare_record_gaps(self):
        self.assertAlmostEqual(require_same_evaluation(gap_record('3034_3',18),gap_record('3034_3',17)),-1)

    def test_each_mismatched_axis_is_rejected(self):
        baseline=gap_record('3034_3',18)
        for field,value in (('dataset','Prime'),('pitch','BaseballThrow_001'),('athlete',999),('session',999),
                            ('target','tracked_ball'),('event_or_window','977-979'),('metric','future_speed_mae'),('units','mph')):
            with self.subTest(field=field):
                other=copy.deepcopy(baseline);other[field]=value
                with self.assertRaises(ValueError):require_same_evaluation(baseline,other)

    def test_missing_unknown_and_nonfinite_are_not_zero_error(self):
        before=gap_record('3034_3',18)
        for value in (None,float('nan'),float('inf')):
            with self.assertRaises(ValueError):require_same_evaluation(before,gap_record('3034_3',value))
        missing=before.copy();del missing['metric']
        with self.assertRaises(ValueError):require_same_evaluation(before,missing)

    def test_different_sources_low_vs_high_error_cannot_be_reported_as_regression(self):
        prime={**gap_record('BaseballThrow_001',.799424),'dataset':'Prime','target':'tracked_ball',
               'event_or_window':'977-979','metric':'future_speed_mae'}
        with self.assertRaises(ValueError):require_same_evaluation(prime,gap_record('3034_3',18.155697))

    def test_publish_preserves_existing_user_folder(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);vault=root/'vault';(vault/'.obsidian').mkdir(parents=True)
            notes=root/'notes';notes.mkdir();(notes/'00.md').write_text('first',encoding='utf-8')
            target=publish(notes,vault);(target/'00.md').write_text('user edit',encoding='utf-8')
            with self.assertRaises(FileExistsError):publish(notes,vault)
            self.assertEqual((target/'00.md').read_text(encoding='utf-8'),'user edit')

    def test_publish_requires_actual_vault(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):publish(d,d)


class SavedAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (OUT/'summary.json').exists():raise unittest.SkipTest('Run same_pitch_audit.py for saved experiment checks')
        cls.result=json.loads((OUT/'summary.json').read_text(encoding='utf-8'))

    def test_complete_grid_and_unknown_ball(self):
        self.assertEqual(len(self.result['records']),5)
        for r in self.result['records']:
            self.assertIsNone(r['ladder']['direct_ball_speed_kmh'])
            self.assertIsNone(r['ladder']['direct_ball_velocity_mps'])
            self.assertEqual(len(r['models']),3)
            for m in r['models'].values():
                self.assertEqual(set(m['resolutions']),{'1','2'})
                self.assertTrue(all(x['state_resets']==0 for x in m['resolutions'].values()))

    def test_speed_attribution_recomputed_independently(self):
        for r in self.result['records']:
            p=r['ladder']['speeds_kmh'];v=r['ladder']['vectors_mps'];m=r['models']['nominal']
            np.testing.assert_allclose(np.asarray(v['fixed_length_wrist'])+v['hand_rotation_component'],v['continuous_assumed_ball'],atol=1e-12)
            self.assertAlmostEqual(np.linalg.norm(v['continuous_assumed_ball'])*3.6,p['continuous_assumed_ball'])
            stages=np.array([r['recorded_speed_kmh'],p['continuous_assumed_ball'],p['free_shoulder_arm_reference'],m['reference_speed_kmh'],m['resolutions']['1']['speed_kmh']])
            np.testing.assert_allclose(np.diff(stages),list(r['attribution_kmh'].values()),atol=1e-10)

    def test_velocity_jacobian_against_finite_difference_positions(self):
        for mode in ('nominal','frozen_skeleton'):
            folder=OUT/'3034_3';model=mujoco.MjModel.from_xml_path(str(folder/(mode+'.xml')))
            z=np.load(folder/(mode+'.npz'));q=z['qpos'][-1];v=z['qvel'][-1]
            eps=1e-7;pa=[];data=mujoco.MjData(model)
            for sign in (-1,1):
                shifted=q.copy();mujoco.mj_integratePos(model,shifted,v,sign*eps)
                data.qpos[:]=shifted;mujoco.mj_forward(model,data)
                pa.append(data.site_xpos[model.site('ball_center').id].copy())
            np.testing.assert_allclose((pa[1]-pa[0])/(2*eps),endpoint(model,q,v),rtol=1e-7,atol=1e-7)

    def test_full_skeleton_geometry_is_frozen_across_pitches(self):
        models=[mujoco.MjModel.from_xml_path(str(OUT/r['pitch']/'frozen_skeleton.xml')) for r in self.result['records']]
        for m in models[1:]:
            for field in ('body_pos','body_mass','body_inertia','site_pos'):
                np.testing.assert_array_equal(getattr(models[0],field),getattr(m,field))

    def test_aggregate_change_error_includes_all_four_pitches(self):
        for mode in ('nominal','frozen_attachment','frozen_skeleton'):
            rows=self.result['records'];pred=np.array([r['models'][mode]['resolutions']['1']['speed_kmh'] for r in rows]);truth=np.array([r['recorded_speed_kmh'] for r in rows])
            expected=np.mean(np.abs((pred[1:]-pred[0])-(truth[1:]-truth[0])))
            self.assertAlmostEqual(expected,self.result['aggregate'][mode]['reused_four_pitch_change_mae_kmh'])

    def test_unvalidated_calibration_is_not_athlete_improvement(self):
        r=self.result
        self.assertFalse(r['actual_athlete_improvement_demonstrated'])
        self.assertFalse(r['same_force_work_intervention_test'])
        self.assertIsNone(r['direct_ball_prediction_error'])
        self.assertFalse(r['adopt_as_performance_model'])


if __name__=='__main__':unittest.main()
