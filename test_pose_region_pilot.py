import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from pose_region_pilot import reconstruct
from prime17_hand_alignment import project


class PoseRegionPilotTests(unittest.TestCase):
    def problem(self, angle=0.):
        s=np.array([0.,0.,4.]);e=s+np.array([.25,.1,.05]);w=e+np.array([.2,-.05,.02])
        q=Rotation.from_rotvec([0.,0.,angle]).as_matrix();ee=s+q@(e-s);ww=s+q@(w-s)
        k=np.array([[500.,0,320.],[0,500.,240.],[0,0,1.]])
        cameras={20:(k,np.eye(3),np.zeros(3)),21:(k,np.eye(3),np.array([-.5,0,0]))};rows=[]
        for c in cameras:
            for name,p in [('l_larm',ee),('l_hand',ww)]:
                u=project(p,cameras[c])[0];rows.append({'camera':c,'segment':name,'box':[u[0]-.5,u[1]-.5,u[0]+.5,u[1]+.5]})
        return s,e,w,rows,cameras

    def test_no_move_when_observations_already_fit(self):
        s,e,w,rows,cameras=self.problem()
        x,_,_=reconstruct(s,e,w,rows,cameras)
        np.testing.assert_allclose(x,np.r_[e,w],atol=1e-7)

    def test_fitted_pose_keeps_both_lengths(self):
        s,e,w,rows,cameras=self.problem(.06)
        x,lengths,info=reconstruct(s,e,w,rows,cameras)
        np.testing.assert_allclose([np.linalg.norm(x[:3]-s),np.linalg.norm(x[3:]-x[:3])],lengths,atol=1e-7)
        self.assertTrue(info['solver_success'])

    def test_heldout_camera_cannot_enter_fit(self):
        s,e,w,rows,cameras=self.problem();rows[0]['camera']=17
        with self.assertRaises(ValueError):reconstruct(s,e,w,rows,cameras)


if __name__=='__main__':unittest.main()
