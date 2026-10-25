import unittest
import numpy as np
import mujoco
from speed_gap_numeric import state_components, speed
from coupled_payload import model_for


class SpeedGapTests(unittest.TestCase):
    def test_translation_only_moves_wrist_and_ball_identically(self):
        m=model_for([.3,.25],.001);d=mujoco.MjData(m)
        v=np.zeros(m.nv);v[:3]=[1,2,-3]
        r=state_components(m,d.qpos,v)
        np.testing.assert_allclose(r['ball_vector_mps'],v[:3],atol=1e-12)
        np.testing.assert_allclose(r['rotation_vector_mps'],0,atol=1e-12)

    def test_rotating_offset_obeys_cross_product_and_finite_difference(self):
        m=model_for([.3,.25],.001);d=mujoco.MjData(m)
        v=np.zeros(m.nv);v[10]=10
        r=state_components(m,d.qpos,v)
        np.testing.assert_allclose(r['wrist_vector_mps'],0,atol=1e-12)
        self.assertAlmostEqual(r['ball_speed_kmh'],1.5*3.6)
        q=d.qpos.copy();eps=1e-6;points=[]
        for sign in [-1,1]:
            d.qpos[:]=q;mujoco.mj_integratePos(m,d.qpos,v,sign*eps)
            mujoco.mj_forward(m,d);points.append(d.site_xpos[m.site('ball_center').id].copy())
        np.testing.assert_allclose((points[1]-points[0])/(2*eps),r['ball_vector_mps'],atol=1e-9)

    def test_component_norms_do_not_add(self):
        self.assertNotAlmostEqual(speed(np.array([1.,0,0]))+speed(np.array([0.,1,0])),speed(np.array([1.,1,0])))

    def test_kinematic_mapping_independent_of_generic_inertia(self):
        results=[]
        for scale in [.8,1.,1.2]:
            m=model_for([.3,.25],.001,hand_mass_scale=scale);d=mujoco.MjData(m)
            results.append(state_components(m,d.qpos,np.linspace(-2,2,m.nv))['ball_vector_mps'])
        np.testing.assert_allclose(results,[results[0]]*3,atol=1e-12)

    def test_arbitrary_pose_jacobian_agrees_with_central_difference(self):
        rng=np.random.default_rng(20261026)
        m=model_for([.3,.25],.001);d=mujoco.MjData(m)
        for _ in range(6):
            d=mujoco.MjData(m)
            mujoco.mj_integratePos(m,d.qpos,rng.normal(size=m.nv),.4)
            q=d.qpos.copy();v=rng.normal(size=m.nv)
            result=state_components(m,q,v);points=[]
            for sign in [-1,1]:
                d.qpos[:]=q;mujoco.mj_integratePos(m,d.qpos,v,sign*1e-6)
                mujoco.mj_forward(m,d);points.append(d.site_xpos[m.site('ball_center').id].copy())
            np.testing.assert_allclose((points[1]-points[0])/2e-6,result['ball_vector_mps'],atol=2e-9)


if __name__=='__main__':unittest.main()
