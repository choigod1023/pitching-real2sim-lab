import unittest
import mujoco
import numpy as np
from muscle_capacity import reduced_coordinates, muscle_force_map


class ReducedDynamicsTests(unittest.TestCase):
    def model(self):
        return mujoco.MjModel.from_xml_string('''<mujoco><option gravity="0 0 0"/>
        <worldbody><body><joint name="a" axis="0 0 1"/><geom type="capsule" fromto="0 0 0 1 0 0" size=".03"/>
        <site name="s0" pos=".1 .1 0"/><body pos="1 0 0"><joint name="b" axis="0 0 1"/>
        <geom type="capsule" fromto="0 0 0 1 0 0" size=".03"/><site name="s1" pos=".8 .1 0"/>
        </body></body></worldbody><equality><joint joint1="b" joint2="a" polycoef="0 .5 0 0 0"/></equality>
        <tendon><spatial name="t"><site site="s0"/><site site="s1"/></spatial></tendon>
        <actuator><muscle tendon="t" lengthrange=".1 2" force="100"/></actuator></mujoco>''')

    def test_equality_virtual_work_projection(self):
        m = self.model()
        S, offset, independent = reduced_coordinates(m)
        np.testing.assert_allclose(S, [[1], [.5]])
        d = mujoco.MjData(m)
        d.qpos[:] = offset + S[:, 0] * .4
        d.qvel[:] = S[:, 0] * .1
        mujoco.mj_forward(m, d)
        d.qacc[:] = S[:, 0] * 2
        mujoco.mj_inverse(m, d)
        M = np.zeros((m.nv, m.nv))
        mujoco.mj_fullM(m, d, M)
        np.testing.assert_allclose(S.T @ d.qfrc_constraint, 0, atol=1e-10)
        np.testing.assert_allclose(S.T @ d.qfrc_inverse,
                                  S.T @ (M @ d.qacc + d.qfrc_bias - d.qfrc_passive), atol=1e-10)

    def test_affine_muscle_torque_reconstruction(self):
        m = self.model()
        d = mujoco.MjData(m)
        d.qpos[:] = [.4, .2]
        mujoco.mj_forward(m, d)
        R, f0, delta = muscle_force_map(m, d)
        d.act[:] = .37
        mujoco.mj_fwdActuation(m, d)
        np.testing.assert_allclose(d.actuator_force, f0 + .37 * delta, atol=1e-10)
        np.testing.assert_allclose(d.qfrc_actuator, R.T @ (f0 + .37 * delta), atol=1e-10)


if __name__ == '__main__':
    unittest.main()
