import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from grip_identifiability import audit_map,cross_matrix,BASE,RADIUS,velocity_map
from coupled_payload import model_for
import mujoco


class GripIdentificationTests(unittest.TestCase):
    def test_exact_null_direction_changes_position_not_velocity(self):
        r=Rotation.from_rotvec([.3,-.1,.2]).as_matrix();omega=np.array([12.,-4.,8.])
        result=audit_map(np.array([20.,3.,1.]),cross_matrix(omega)@r,r,omega)
        self.assertAlmostEqual(result['null_axis_position_difference_mm'],10.)
        self.assertLess(result['null_axis_velocity_difference_mps'],1e-14)

    def test_sphere_bound_covers_arbitrary_directions(self):
        r=np.eye(3);w=np.array([0.,0.,100.]);a=cross_matrix(w);v=np.array([30.,0.,0.])
        result=audit_map(v,a,r,w)
        directions=np.random.default_rng(42).normal(size=(1000,3));directions/=np.linalg.norm(directions,axis=1)[:,None]
        speeds=np.linalg.norm(v+(BASE+directions*RADIUS)@a.T,axis=1)*3.6
        self.assertLessEqual(speeds.max(),result['all_radius_ball_speed_upper_bound_kmh'])

    def test_velocity_affine_map_matches_independent_site_jacobian(self):
        m=model_for([.3,.25],.001);d=mujoco.MjData(m)
        d.qpos[:]=m.qpos0;d.qvel[:]=np.linspace(-2,3,m.nv)
        v,a,_,_=velocity_map([.3,.25],.001,d.qpos,d.qvel)
        for grip in ([.14,.01,-.01],[.16,-.01,.01]):
            n=model_for([.3,.25],.001,grip=grip);e=mujoco.MjData(n)
            e.qpos[:]=d.qpos;e.qvel[:]=d.qvel;mujoco.mj_forward(n,e)
            jp,jr=np.zeros((3,n.nv)),np.zeros((3,n.nv))
            mujoco.mj_jacSite(n,e,jp,jr,n.site('ball_center').id)
            np.testing.assert_allclose(jp@e.qvel,v+a@grip,atol=1e-12)
