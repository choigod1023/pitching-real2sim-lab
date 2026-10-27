import tempfile,unittest
from pathlib import Path
import numpy as np
import mujoco
from noise_structure_audit import mobile_model

class MobileOriginTests(unittest.TestCase):
    def test_preserves_mass_and_tracks_rotated_parent_target(self):
        xml='''<mujoco><worldbody><body name="torso" quat="0.9238795325 0 0 0.3826834324"><geom size=".1" mass="10"/>
        <body name="upper" pos=".2 0 0"><joint name="pitch_shoulder" type="ball"/><geom size=".1" mass="2"/><site name="shoulder_jc"/></body>
        <body name="glove_upper" pos="-.2 0 0"><joint name="glove_shoulder" type="ball"/><geom size=".1" mass="2"/><site name="glove_shoulder_jc"/></body>
        </body></worldbody></mujoco>'''
        m=mujoco.MjModel.from_xml_string(xml);q=np.tile(m.qpos0,(3,1))
        points={'shoulder_jc':np.tile([.22,.18,.03],(3,1)),
                'glove_shoulder_jc':np.tile([-.1,-.2,.05],(3,1))}
        with tempfile.TemporaryDirectory() as tmp:
            mm,qq,_=mobile_model(m,q,points,Path(tmp)/'model.xml')
            self.assertAlmostEqual(m.body_mass.sum(),mm.body_mass.sum())
            self.assertEqual(mm.nv,m.nv+6)
            d=mujoco.MjData(mm)
            for row in qq:
                d.qpos[:]=row;mujoco.mj_forward(mm,d)
                for name in points:np.testing.assert_allclose(d.site_xpos[mm.site(name).id],points[name][0],atol=1e-9)

if __name__=='__main__':unittest.main()
