import unittest
import numpy as np
import mujoco
from whole_chain_audit import replace_components
from baseline_ground_balance import wrench_at_root


class TestChain(unittest.TestCase):
    def test_vector_replacement_closure_and_order_averaging(self):
        base=np.array([[1.,0,0],[0,2.,0],[0,0,3.],[1.,2.,3.]])
        observed=base+np.array([[.1,0,0],[0,-.3,0],[0,0,.2],[-.1,.2,0]])
        hand=np.array([1.,1.,0])
        values,phi=replace_components(base,observed,hand)
        self.assertAlmostEqual(sum(phi),values[(1,1,1,1)]-values[(0,0,0,0)],places=12)
        permutation=[2,0,3,1]
        _,other=replace_components(base[permutation],observed[permutation],hand)
        np.testing.assert_allclose(other,np.array(phi)[permutation],atol=1e-12)

    def test_opposing_component_changes_can_cancel(self):
        base=np.ones((4,3));observed=base.copy();observed[0,0]+=2;observed[1,0]-=2
        values,phi=replace_components(base,observed,np.zeros(3))
        self.assertAlmostEqual(sum(phi),0.)
        self.assertAlmostEqual(values[(0,0,0,0)],values[(1,1,1,1)])

    def test_global_plate_wrench_includes_lever_arm(self):
        m=mujoco.MjModel.from_xml_string('<mujoco><worldbody><body name="pelvis"><freejoint/><geom type="sphere" size=".1" mass="1"/></body></worldbody></mujoco>')
        d=mujoco.MjData(m);d.qpos[:3]=[.1,.2,.3];mujoco.mj_forward(m,d)
        actual=wrench_at_root(m,d,[[0.,2.,0.]],[[0.,0.,3.]],[[1.1,.2,.3]])
        np.testing.assert_allclose(actual,[0,2,0,0,0,5],atol=1e-12)


if __name__=='__main__':unittest.main()
