import unittest
import numpy as np
import pandas as pd
from load_matched_pitching import pair_records,clustered_summary,POI,LOADS
from render_load_matched import make_track,replay,NAMES


def row(pid,speed,user=1,**kw):
    r=dict(user=user,session=user,pitch_type='FF',session_pitch=pid,speed_kmh=speed,
           elbow_varus_moment=100.,shoulder_internal_rotation_moment=100.,
           arm_generation_j=20.,partial_six_joint_generation_j=100.,stride_cm=130.)
    r.update({k:0. for k in POI});r.update(kw);return r


class LoadMatchedTests(unittest.TestCase):
    def test_true_proxy_matched_pair_and_delta_direction(self):
        slow=row('s',130);fast=row('f',132,elbow_varus_moment=99,shoulder_internal_rotation_moment=98,arm_generation_j=19)
        p=pair_records(pd.DataFrame([fast,slow])).iloc[0]
        self.assertTrue(p.eligible);self.assertEqual((p.slow,p.fast),('s','f'));self.assertEqual(p.speed_gain_kmh,2)
        self.assertEqual(p.elbow_varus_moment,-1)

    def test_increased_load_or_generation_and_subthreshold_gain_rejected(self):
        for kw in [dict(elbow_varus_moment=101),dict(shoulder_internal_rotation_moment=101),dict(arm_generation_j=21)]:
            self.assertFalse(pair_records(pd.DataFrame([row('s',130),row('f',132,**kw)])).iloc[0].eligible)
        self.assertFalse(pair_records(pd.DataFrame([row('s',130),row('f',130.9)])).iloc[0].eligible)

    def test_no_comparisons_across_athletes_sessions_or_pitch_types(self):
        for fast in [row('f',132,user=2),row('f',132,session=2),row('f',132,pitch_type='SL')]:
            self.assertTrue(pair_records(pd.DataFrame([row('s',130),fast])).empty)

    def test_duplicate_ids_rejected_and_equal_speed_excluded(self):
        with self.assertRaises(ValueError):pair_records(pd.DataFrame([row('s',130),row('s',132)]))
        self.assertTrue(pair_records(pd.DataFrame([row('s',130),row('f',130)])).empty)

    def test_statistics_weight_athletes_not_reused_pairs(self):
        p=pd.DataFrame(dict(user=[1,1,1,2],feature=[1,1,1,9]))
        out=clustered_summary(p,['feature'])[0]
        self.assertEqual(out['mean_delta'],5);self.assertNotEqual(out['mean_delta'],p.feature.mean())
        self.assertEqual(out['n_athletes'],2)

    def test_subsample_timing_not_given_a_direction(self):
        p=pd.DataFrame(dict(user=[1,2,3],test_ms=[2.,-2.,4.]))
        out=clustered_summary(p,['test_ms'])[0]
        self.assertEqual(out['unresolved_athletes'],2);self.assertEqual(out['positive_athletes'],1)

    def test_nonfinite_comparison_statistics_rejected(self):
        with self.assertRaises(ValueError):clustered_summary(pd.DataFrame(dict(user=[1],feature=[np.nan])),['feature'])
        with self.assertRaises(ValueError):pair_records(pd.DataFrame([row('s',130),row('f',132,arm_generation_j=np.nan)]))

    def test_overlay_preserves_height_and_elapsed_motion_after_fixed_xy_alignment(self):
        t=np.arange(361)/360
        columns=dict(time=t,fp_10_time=np.full(len(t),.4),BR_time=np.full(len(t),.56))
        for n in NAMES:
            columns[n+'_x']=t+2;columns[n+'_y']=np.full(len(t),3.);columns[n+'_z']=np.full(len(t),1.)
        angles=pd.DataFrame(dict(time=t,lead_knee_angle_x=t*10,torso_angle_z=t*20,torso_pelvis_angle_z=t*30))
        track=make_track(pd.DataFrame(columns),angles)
        at_fp=replay(track,.4);later=replay(track,.5)
        np.testing.assert_allclose(at_fp['wrist_jc'],[0,0,1],atol=1e-12)
        np.testing.assert_allclose(later['wrist_jc']-at_fp['wrist_jc'],[.1,0,0],atol=1e-12)
        with self.assertRaises(ValueError):replay(track,1.01)


if __name__=='__main__':unittest.main()
