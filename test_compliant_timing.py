import unittest
import numpy as np
from compliant_timing import coupling,taper,budget


class CompliantTests(unittest.TestCase):
    def test_internal_force_power_and_angular_momentum(self):
        a=np.array([.1,.2,.3]);b=np.array([.11,.19,.32]);av=np.array([2.,-1.,3.]);bv=np.array([4.,5.,2.])
        f,energy,loss=coupling(a,b,av,bv,10000,2)
        np.testing.assert_allclose(np.cross(a,-f)+np.cross(b,f),0,atol=1e-12)
        elastic_rate=10000*(b-a)@(bv-av)
        self.assertAlmostEqual(float(f@(bv-av))+elastic_rate,-loss,places=10)
        self.assertGreater(energy,0);self.assertGreaterEqual(loss,0)

    def test_release_only_removes_energy(self):
        scale,rate=taper(.029,.030)
        self.assertAlmostEqual(scale,.5);self.assertLess(rate,0)
        f,e,loss=coupling(np.zeros(3),[.01,0,0],np.zeros(3),np.zeros(3),10000*scale,2*scale)
        self.assertGreater(-.5*10000*rate*.01**2,0)
        self.assertEqual(taper(.03,.03),(0.,0.))
        np.testing.assert_array_equal(coupling([0,0,0],[0,0,0],[1,0,0],[2,0,0],10000,2)[0],[0,0,0])

    def test_gate_rejects_one_component_increase_despite_total_decrease(self):
        keys=['work_positive_j','work_negative_j','peak_per_dof','group_peaks','translation_work_pm_j','coupling_peak_force_n','coupling_loss_j']
        base={k:[10.,10.] for k in keys};case={k:[10.,10.] for k in keys}
        self.assertEqual(budget(base,case),{})
        case['work_positive_j']=[11.,0.]
        self.assertIn('work_positive_j',budget(base,case))


if __name__=='__main__':unittest.main()
