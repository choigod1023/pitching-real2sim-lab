import unittest
import numpy as np
from signed_video_error import describe, linear_speed


class TestSignedError(unittest.TestCase):
    def test_bias_not_absolute_error(self):
        x=describe([-3.,-1.,1.])
        self.assertAlmostEqual(x['mean_signed'],-1.)
        self.assertAlmostEqual(x['sample_sd'],2.)
        self.assertAlmostEqual(x['mae'],5/3)

    def test_nonuniform_frames_preserve_velocity(self):
        frames=np.array([980,981,985])
        points=np.array([1.,2.,3.])+frames[:,None]/360*np.array([3.,4.,0.])
        self.assertAlmostEqual(linear_speed(frames,points),18.,places=10)

    def test_invalid_spread_rejected(self):
        for x in [[1.],[1.,float('nan')]]:
            with self.assertRaises(ValueError):describe(x)


if __name__=='__main__':unittest.main()
