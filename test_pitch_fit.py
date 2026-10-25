import unittest
from pathlib import Path
import numpy as np
from pitch_fit import chest_frames, ArmMap
from muscle_model_check import load_arm


class FrameTests(unittest.TestCase):
    def test_frame_orthonormal_and_translation_invariant(self):
        r=np.array([[0.,0.,1.]]);l=-r;prox=np.array([[0.,1.,0.]]);dist=np.zeros((1,3))
        b=chest_frames(r,l,prox,dist)
        np.testing.assert_allclose(b[0],np.eye(3))
        np.testing.assert_allclose(chest_frames(r+5,l+5,prox+5,dist+5),b)
        self.assertAlmostEqual(np.linalg.det(b[0]),1.)

    def test_degenerate_frame_rejected(self):
        z=np.zeros((1,3))
        with self.assertRaises(ValueError): chest_frames(z,z,z,z)


@unittest.skipUnless(Path('artifacts/vendor/myo_sim').is_dir(),'install pinned model to run integration checks')
class ArmFitTests(unittest.TestCase):
    def test_synthetic_recovery_and_constraints(self):
        model,_=load_arm(Path('artifacts/vendor'));mapping=ArmMap(model)
        known=np.array([.4,1.2,.5,1.5])
        target=mapping.set_pose(known).copy()
        _,error,ok=mapping.fit(target)
        self.assertTrue(ok);self.assertLess(error.max(),.05)
        for dep,src,offset,slope in mapping.constraints:
            self.assertAlmostEqual(mapping.data.qpos[dep],offset+slope*mapping.data.qpos[src])
        self.assertLess(mapping.upper[1],model.joint('shoulder_elv_r').range[1])

    def test_temporal_solver_preserves_constant_feasible_motion(self):
        model,_=load_arm(Path('artifacts/vendor'));mapping=ArmMap(model)
        known=np.array([.4,1.2,.5,1.5])
        targets=np.repeat(mapping.set_pose(known)[None,:,:],3,axis=0)
        fitted,ok=mapping.fit_sequence(targets,np.tile(known,(3,1)),10.)
        self.assertTrue(ok)
        np.testing.assert_allclose(fitted,np.tile(known,(3,1)),atol=1e-6)
