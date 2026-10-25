import unittest
from pathlib import Path
import numpy as np
from kinematics import Trial, Skeleton, frame_rotation


class KinematicChecks(unittest.TestCase):
    def test_frame_is_right_handed_and_handles_parallel_hint(self):
        for z,y in [([0,0,1],[0,0,1]),([1,2,3],[0,1,0]),([0,0,-1],[0,1,0])]:
            matrix=frame_rotation(z,y).as_matrix()
            np.testing.assert_allclose(matrix.T@matrix,np.eye(3),atol=1e-12)
            self.assertAlmostEqual(np.linalg.det(matrix),1.)
            np.testing.assert_allclose(matrix[:,2],np.array(z)/np.linalg.norm(z),atol=1e-12)

    def test_zero_segment_rejected(self):
        with self.assertRaises(ValueError):frame_rotation([0,0,0],[0,1,0])

    def test_recorded_trial_has_fixed_model_lengths_and_correct_directions(self):
        root=Path(__file__).resolve().parent.parent/'openbiomechanics/baseball_pitching/data/c3d'
        files=sorted(p for p in root.rglob('*.c3d') if 'model' not in p.name.lower())
        if not files:self.skipTest('External OBP data not downloaded')
        trial=Trial.read(files[0]); skeleton=Skeleton(trial)
        for index in [0,trial.count//2,trial.count-1]:
            skeleton.pose(index)
            self.assertTrue(np.isfinite(skeleton.data.qpos).all())
            for side in 'LR':
                for name,a,b in [('upperarm','shoulder','elbow'),('forearm','elbow','wrist'),('thigh','hip','knee'),('shin','knee','ankle')]:
                    start=skeleton.data.site(side+a).xpos
                    end=skeleton.data.site(side+b).xpos
                    self.assertAlmostEqual(np.linalg.norm(end-start),skeleton.lengths[side+name],places=10)
                    actual=trial.joints[side+b][index]-trial.joints[side+a][index]
                    np.testing.assert_allclose((end-start)/np.linalg.norm(end-start),actual/np.linalg.norm(actual),atol=1e-10)
            baseline=skeleton.data.site('Rwrist').xpos.copy()
            skeleton.pose(index,1.3)
            np.testing.assert_allclose(skeleton.data.site('Rwrist').xpos-baseline,[0,1.3,0],atol=1e-10)


if __name__=='__main__':unittest.main()
