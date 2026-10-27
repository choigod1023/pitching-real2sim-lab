import unittest
import numpy as np
from follow_path import follow_offset
from follow_contacts import contacts


class FollowPathTests(unittest.TestCase):
    def test_disabled_and_bounded(self):
        for x in [[1,2,-3],[0,0,0]]:
            s,v=follow_offset(x,[1,2,3],False);np.testing.assert_equal(s,0);np.testing.assert_equal(v,0)
            s,v=follow_offset(x,[1,2,3]);self.assertLessEqual(max(abs(s)),.005)

    def test_velocity_matches_position_derivative(self):
        e=np.array([.003,-.014,.0002]);v=np.array([.4,.8,-.2]);h=1e-7
        _,rate=follow_offset(e,v)
        numeric=(follow_offset(e+h*v,v)[0]-follow_offset(e-h*v,v)[0])/(2*h)
        np.testing.assert_allclose(rate,numeric,atol=1e-9)

    def test_moving_pad_power_matches_elastic_and_dissipation(self):
        x=np.array([.0001,-.0002,.00005]);v=np.array([.02,-.01,.003]);c=np.zeros(3);hv=np.array([.01,.005,0])
        shift,rate=follow_offset(x-c,v-hv)
        r=contacts(x,v,np.zeros(3),c+shift,hv+rate,np.zeros(3),np.eye(3),0,0)
        def elastic(t):
            xx=x+t*v;cc=c+t*hv;ss,_=follow_offset(xx-cc,v-hv)
            return contacts(xx,v,np.zeros(3),cc+ss,hv+rate,np.zeros(3),np.eye(3),0,0)[3]
        edot=(elastic(1e-7)-elastic(-1e-7))/(2e-7)
        self.assertAlmostEqual(float(r[0]@v)+edot,float(r[2].sum())-r[4],places=7)


if __name__=='__main__':unittest.main()
