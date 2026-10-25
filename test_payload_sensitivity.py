import unittest
import numpy as np
from coupled_payload import model_for,inverse_and_rollout,fit_pose


class SensitivityTests(unittest.TestCase):
    def pose(self):
        p=np.array([[[0.,0.,1.5],[0.,0.,1.2],[.2,0.,1.05]]])
        return np.repeat(fit_pose(p,np.array([[.1,.05,1.125]]),np.eye(3)[None])[0],51,axis=0)

    def test_fixed_nominal_loads_reproduce_nominal_run(self):
        m=model_for([.3,.25],.001);q=self.pose()
        first,loads=inverse_and_rollout(m,q,.001)
        second,_=inverse_and_rollout(m,q,.001,applied_loads=loads)
        self.assertEqual(first['ball_point_rmse_mm'],second['ball_point_rmse_mm'])
        self.assertEqual(second['load_policy'],'fixed_supplied')

    def test_extra_hand_mass_requires_extra_weight_support(self):
        q=self.pose()
        _,base=inverse_and_rollout(model_for([.3,.25],.001),q,.001)
        _,heavy=inverse_and_rollout(model_for([.3,.25],.001,hand_mass_scale=1.2),q,.001)
        self.assertAlmostEqual(heavy[0,2]-base[0,2],.2*.482044*9.81,places=10)

    def test_invalid_fixed_forces_rejected(self):
        with self.assertRaises(ValueError):
            inverse_and_rollout(model_for([.3,.25],.001),self.pose(),.001,applied_loads=np.zeros((3,2)))
