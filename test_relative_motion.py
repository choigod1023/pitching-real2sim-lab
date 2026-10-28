"""Independent analytical motions and input/holdout checks for hand-ball decomposition."""
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from video_track.relative_motion import analyze, motion_curves, window_speed
from video_track.theia import load_hand, load_hand_pose


def synthetic(omega=0.0, sliding=False):
    frames = np.arange(956, 996)
    t = (frames-956)/360
    hand = np.outer(t, [2.0, -1.0, 0.5])
    rot = Rotation.from_rotvec(np.outer(t, [0.0, 0.0, omega])).as_matrix()
    r = np.tile([0.15, 0.04, 0.02], (len(t), 1))
    if sliding:
        r += np.outer(t, [1.0, -0.5, 0.2])
    ball = hand + np.einsum('nij,nj->ni', rot, r)
    return frames, hand, rot, ball, r


class RelativeMotionTests(unittest.TestCase):
    def test_rigid_translation_has_no_relative_or_rotation_velocity(self):
        f, h, R, b, r = synthetic()
        summary, _, state = analyze(f, h, R, b)
        np.testing.assert_allclose(state['local_ball_m'], r, atol=1e-12)
        np.testing.assert_allclose(state['relative_velocity_local_mps'], 0, atol=1e-10)
        self.assertLess(summary['fixed_heldout_position_rmse_mm'], 1e-9)
        self.assertLess(summary['fixed_heldout_speed_mae_kmh'], 1e-9)

    def test_rotating_rigid_grip_checks_angular_velocity_frame_and_cross_sign(self):
        f, h, R, b, r = synthetic(omega=20.0)
        _, _, state = analyze(f, h, R, b)
        lever = b-h
        expected = np.cross(np.tile([0, 0, 20.0], (len(f), 1)), lever)
        np.testing.assert_allclose(state['rotation_velocity_mps'], expected, atol=1e-10)
        np.testing.assert_allclose(state['ball_velocity_mps'][3:-3], expected[3:-3]+[2,-1,.5], atol=1e-5)
        np.testing.assert_allclose(state['relative_velocity_local_mps'][3:-3], 0, atol=1e-5)

    def test_sliding_ball_recovers_known_relative_velocity(self):
        f, h, R, b, r = synthetic(sliding=True)
        _, rows, state = analyze(f, h, R, b)
        np.testing.assert_allclose(state['relative_velocity_local_mps'], np.tile([1,-.5,.2], (len(f), 1)), atol=1e-10)
        for row in rows:
            self.assertAlmostEqual(sum(row['signed_speed_components_kmh'].values()), row['observed_spline_speed_kmh'], places=9)

    def test_world_frame_change_preserves_local_motion(self):
        f, h, R, b, _ = synthetic(omega=20, sliding=True)
        _, _, original = analyze(f,h,R,b)
        Q = Rotation.from_euler('xyz', [.3,-.6,1.1]).as_matrix()
        _, _, other = analyze(f,h@Q.T+[4,-3,9], np.einsum('ij,njk->nik',Q,R), b@Q.T+[4,-3,9])
        np.testing.assert_allclose(other['local_ball_m'],original['local_ball_m'],atol=1e-12)
        np.testing.assert_allclose(other['relative_velocity_local_mps'],original['relative_velocity_local_mps'],atol=1e-9)

    def test_heldout_ball_does_not_change_fitted_fixed_grip_or_velocity(self):
        f,h,R,b,_=synthetic(omega=20,sliding=True)
        s1,_,a=analyze(f,h,R,b)
        changed=b.copy();changed[f>=964]+=[.01,.02,-.01]
        s2,_,z=analyze(f,h,R,changed)
        np.testing.assert_array_equal(s1['fixed_local_vector_mm'],s2['fixed_local_vector_mm'])
        np.testing.assert_array_equal(a['fixed_ball_m'],z['fixed_ball_m'])
        np.testing.assert_array_equal(a['fixed_ball_velocity_mps'],z['fixed_ball_velocity_mps'])
        self.assertNotEqual(s1['fixed_heldout_position_rmse_mm'],s2['fixed_heldout_position_rmse_mm'])

    def test_independent_difference_of_relative_curve_matches_formula(self):
        f,h,R,b,_=synthetic(omega=20,sliding=True)
        summary,_,_=analyze(f,h,R,b)
        self.assertLess(summary['relative_derivative_fd_max_mps'],1e-7)

    def test_rejects_missing_reflected_and_outside_inputs(self):
        f,h,R,b,_=synthetic()
        with self.assertRaises(ValueError):analyze(np.delete(f,4),np.delete(h,4,0),np.delete(R,4,0),np.delete(b,4,0))
        bad=R.copy();bad[:,:,0]*=-1
        with self.assertRaises(ValueError):analyze(f,h,bad,b)
        bad=b.copy();bad[10,0]=np.nan
        with self.assertRaises(ValueError):analyze(f,h,R,bad)
        with self.assertRaises(ValueError):analyze(f,h,R,b,eval_range=(960,973))
        curve=motion_curves((f-f[0])/360,h,R,b)
        with self.assertRaises(ValueError):curve(-.01)
        with self.assertRaises(ValueError):window_speed((f-f[0])/360,b,4)


class HandPoseTests(unittest.TestCase):
    def source(self):
        matrices=np.repeat(np.eye(4)[:,:,None],6,axis=2)
        matrices[:3,3,:]=np.array([[100],[200],[300]])
        source=dict(parameters={'ROTATION':{'LABELS':{'value':['l_hand_4X4']},'RATE':{'value':[360.]}}},
                    data={'rotations':matrices[:,:,None,:]})
        q=Rotation.from_euler('z',.4).as_matrix()
        alignment=dict(rotation=q.tolist(),translation=[1,2,3],scale=.001,trial='throw')
        return source,alignment,q

    def test_pose_scales_positions_only_and_preserves_legacy_origin(self):
        src,al,q=self.source()
        with patch('video_track.theia.ezc3d.c3d',return_value=src),patch.object(Path,'read_text',return_value=json.dumps(al)):
            pose=load_hand_pose('throw.c3d','align.json');origin=load_hand('throw.c3d','align.json')
            h,R=pose(1,[0,1,2]);np.testing.assert_allclose(h,np.tile(q@np.array([.1,.2,.3])+[1,2,3],(3,1)))
            np.testing.assert_allclose(R,np.tile(q,(3,1,1)))
            np.testing.assert_array_equal(origin(1,[0,1,2]),h)
            for offset,frames in [(-1,[0]),(1,[5]),(0,[1.5]),(.5,[1])]:
                with self.assertRaises(ValueError):pose(offset,frames)

    def test_mismatched_trial_rate_and_alignment_rejected(self):
        src,al,_=self.source()
        with patch('video_track.theia.ezc3d.c3d',return_value=src),patch.object(Path,'read_text',return_value=json.dumps(al)):
            with self.assertRaises(ValueError):load_hand_pose('other.c3d','align.json')
            with self.assertRaises(ValueError):load_hand_pose('throw.c3d','align.json',fps=240)
        al['rotation'][0][0]*=2
        with patch('video_track.theia.ezc3d.c3d',return_value=src),patch.object(Path,'read_text',return_value=json.dumps(al)):
            with self.assertRaises(ValueError):load_hand_pose('throw.c3d','align.json')


if __name__=='__main__':unittest.main()
