import unittest
import numpy as np
import opensim as o
from anatomical_shoulder_fit import OUT,arr,register
from anatomical_muscle_screen import ov

class AnatomicalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        o.Logger.setLevelString('error');cls.m=o.Model(str(OUT/'scaled_source.osim'))
    def test_registration_is_proper_rigid_transform(self):
        a=np.array([[0,0,0],[1,0,0],[0,1,0.]])
        r,t=register(a,a+[2,3,4]);np.testing.assert_allclose(a@r+t,a+[2,3,4],atol=1e-12);self.assertAlmostEqual(np.linalg.det(r),1.)
    def test_quaternion_velocity_conversion_round_trip(self):
        m=self.m;s=m.initSystem();m.realizePosition(s);mat=m.getMatterSubsystem()
        u=np.arange(s.getNU())*.01;qdot=o.Vector();back=o.Vector()
        mat.multiplyByN(s,False,ov(u),qdot);mat.multiplyByNInv(s,False,qdot,back)
        np.testing.assert_allclose(arr(back),u,atol=1e-10)
        self.assertNotEqual(s.getNQ(),s.getNU())
    def test_path_velocity_matches_finite_difference(self):
        m=self.m;s=m.initSystem();u=np.linspace(-.1,.1,s.getNU());s.setU(ov(u));m.realizeVelocity(s)
        muscle=m.getMuscles().get(0);path=muscle.getGeometryPath();rate=path.getLengtheningSpeed(s)
        q=arr(s.getQ());dq=o.Vector();m.getMatterSubsystem().multiplyByN(s,False,ov(u),dq);dq=arr(dq)
        eps=1e-6;length=[]
        for sign in [-1,1]:
            s.setQ(ov(q+sign*eps*dq));m.realizePosition(s);length.append(path.getLength(s))
        self.assertAlmostEqual((length[1]-length[0])/(2*eps),rate,places=5)
    def test_source_constraint_and_muscles_retained(self):
        self.assertEqual(self.m.getMuscles().getSize(),33)
        self.assertEqual(self.m.getJointSet().get('scapulothoracic').getConcreteClassName(),'ScapulothoracicJoint')
        s=self.m.initSystem();self.m.realizePosition(s)
        self.assertLess(np.linalg.norm(arr(s.getQErr())),1e-8)

if __name__=='__main__':unittest.main()
