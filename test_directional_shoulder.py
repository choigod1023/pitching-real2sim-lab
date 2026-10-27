import unittest
import numpy as np
from anatomical_shoulder_fit import OUT as BASE
from directional_shoulder_fit import OUT

class DirectionalShoulderTests(unittest.TestCase):
    def test_skin_frames_are_proper_rotations_and_samples_align(self):
        for pitch in ['3034_1','3034_2','3034_3','3034_4','3034_5']:
            z=np.load(OUT/f'{pitch}_directions.npz');base=np.load(BASE/f'{pitch}_input.npz');r=z['frames']
            np.testing.assert_allclose(z['time'],base['time'],atol=1e-12)
            np.testing.assert_allclose(np.linalg.det(r),1.,atol=1e-12)
            np.testing.assert_allclose(np.einsum('nbji,nbjk->nbik',r,r),np.broadcast_to(np.eye(3),r.shape),atol=1e-12)
    def test_no_muscle_parameters_changed_for_direction_fit(self):
        self.assertEqual((BASE/'scaled_source.osim').read_bytes(),(OUT/'scaled_source.osim').read_bytes())

if __name__=='__main__':unittest.main()
