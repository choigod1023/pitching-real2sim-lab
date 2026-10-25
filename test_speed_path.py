import unittest
import numpy as np
from scipy.spatial.transform import Rotation,RotationSpline
from speed_path import retarget_velocity,direction_derivative,rotation_velocity


class SpeedPathTests(unittest.TestCase):
    def test_fixed_lengths_translation_preserved(self):
        points=np.array([[0.,0.,0.],[0.,0.,-.3],[.2,0.,-.45]])
        velocity=np.tile([2.,3.,4.],(3,1))
        np.testing.assert_allclose(retarget_velocity(points,velocity,[.3,.25]),[2.,3.,4.])

    def test_radial_length_change_removed(self):
        np.testing.assert_allclose(direction_derivative(np.array([2.,0.,0.]),np.array([3.,4.,0.])),[0.,2.,0.])

    def test_rotation_component_matches_cross_product(self):
        t=np.linspace(0,1,101);r=RotationSpline(t,Rotation.from_rotvec(np.column_stack([0*t,0*t,2*t])))
        point=r(.5).apply([.15,0.,0.])
        np.testing.assert_allclose(rotation_velocity(r,.5,np.array([.15,0.,0.])),np.cross([0.,0.,2.],point),atol=1e-9)
