import unittest
import numpy as np
from release_calibration import predict_speed, fit_offset, evaluate_heldout


class ReleaseCalibrationTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(72)
        self.v = rng.normal(size=(30,3))*10
        omega = rng.normal(size=(30,3))*50
        self.d = np.zeros((30,3,3))
        for i, (x,y,z) in enumerate(omega):
            self.d[i] = [[0,-z,y],[z,0,-x],[-y,x,0]]
        self.offset = np.array([.025,-.014,.019])
        self.y = predict_speed(self.v, self.d, self.offset)
        self.users = np.repeat(np.arange(6),5)

    def test_known_rigid_velocities_recovered(self):
        fitted = fit_offset(self.v, self.d, self.y, self.users)
        np.testing.assert_allclose(fitted, self.offset, atol=1e-7)

    def test_heldout_labels_cannot_change_fit_or_prediction(self):
        first = evaluate_heldout(self.v,self.d,self.y,self.users,0)
        altered = self.y.copy(); altered[self.users == 0] += 1000
        second = evaluate_heldout(self.v,self.d,altered,self.users,0)
        for key in ['offset','predicted','train_mean','scalar_linear','linear_coefficients']:
            np.testing.assert_array_equal(first[key],second[key])

    def test_known_affine_calibration(self):
        observed = 30.+1.5*np.linalg.norm(self.v,axis=1)*3.6
        result = evaluate_heldout(self.v,self.d,observed,self.users,0)
        np.testing.assert_allclose(result['linear_coefficients'],[30.,1.5],atol=1e-10)
        np.testing.assert_allclose(result['scalar_linear'],observed[self.users==0],atol=1e-10)

    def test_zero_angular_motion_returns_translation(self):
        actual = predict_speed(self.v,np.zeros_like(self.d),self.offset)
        np.testing.assert_allclose(actual,np.linalg.norm(self.v,axis=1)*3.6)


if __name__ == '__main__':
    unittest.main()
