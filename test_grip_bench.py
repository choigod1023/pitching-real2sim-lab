import unittest
import numpy as np
from grip_bench import contacts,DIR,opening,RADIUS,PAD_RADIUS,DEPTH


class GripBenchTests(unittest.TestCase):
    def test_preload_balanced_and_energy_explicit(self):
        f,t,p,e,l,d,forces,active,_,_=contacts(np.zeros(3),np.zeros(3),np.zeros(3),np.zeros(3),np.zeros(3),np.zeros(3),np.eye(3),0,0)
        np.testing.assert_allclose(f,0,atol=1e-11)
        np.testing.assert_allclose(forces,10000/3*.001)
        self.assertAlmostEqual(e,.005);self.assertEqual(active,3)

    def test_finger_motion_energy_accounting(self):
        # Opening separates loaded pads; actuator absorbs stored energy.
        f,t,p,e,l,d,forces,a,rp,pads=contacts(np.zeros(3),np.zeros(3),np.zeros(3),np.zeros(3),np.zeros(3),np.zeros(3),np.eye(3),.0002,.1)
        energy_rate=-10000*(DEPTH-.0002)*.1
        self.assertAlmostEqual(energy_rate,float(p.sum())-l,places=9)
        self.assertLess(p.sum(),0)

    def test_open_has_no_contact_and_smooth_endpoints(self):
        for t,expected in [(0,(0,0)),(.12,(0,0)),(.13,(.012,0)),(.2,(.012,0))]:
            np.testing.assert_allclose(opening(t,.12,.13),expected)
        r=contacts(np.zeros(3),np.zeros(3),np.zeros(3),np.zeros(3),np.zeros(3),np.zeros(3),np.eye(3),.012,0)
        np.testing.assert_allclose(r[0],0);self.assertEqual(r[7],0)


if __name__=='__main__':unittest.main()
