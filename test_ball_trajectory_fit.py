import unittest
import numpy as np
from ball_trajectory_fit import fit_trajectory,triangulate,extract
class TrajectoryTests(unittest.TestCase):
 def test_linear_nonuniform_times(self):
  t=np.array([0,.002,.005,.009]);p=np.array([1,2,3])+t[:,None]*[30000,0,0]
  f=fit_trajectory(t,p);self.assertAlmostEqual(f['midpoint_speed_kmh'],108);self.assertLess(f['position_rmse_mm'],1e-10)
 def test_quadratic(self):
  t=np.arange(6)/360;p=np.column_stack([30000*t,500*t*t,np.zeros(6)])
  f=fit_trajectory(t,p,2);np.testing.assert_allclose(f['acceleration_m_s2'],[0,1,0],atol=1e-8)
 def test_invalid_time(self):
  with self.assertRaises(ValueError):fit_trajectory([0,0],np.zeros((2,3)))
 def test_invalid_values_and_shapes(self):
  for t,p,d in [([0,1],[[0,0,0],[float('nan'),0,0]],1),([0,1],[[0,0],[1,1]],1),([0,1],[[0,0,0],[1,1,1]],0)]:
   with self.assertRaises(ValueError):fit_trajectory(t,p,d)
 def test_component_excludes_remote_blob(self):
  im=np.zeros((100,100,3),dtype=np.uint8);im[47:54,47:54]=200;im[36:43,30:35]=200
  r=extract(im,(50,50),100,'component');np.testing.assert_allclose(r['pixel'],[50,50]);self.assertEqual(r['area_px'],49)
 def test_missing(self):self.assertIsNone(extract(np.zeros((100,100,3)),(50,50),100,'component')['pixel'])
 def test_triangulation(self):
  k=np.diag([800.,800.,1.]);p1=np.column_stack((np.eye(3),np.zeros(3)));p2=np.column_stack((np.eye(3),[-1000,0,0]));x=np.array([100.,200.,4000.]);cams=[(k,p1),(k,p2)]
  uv=[]
  for K,P in cams:q=K@(P@np.r_[x,1]);uv.append(q[:2]/q[2])
  got,res,depth=triangulate(uv,cams);np.testing.assert_allclose(got,x,atol=1e-8);self.assertLess(max(res),1e-8);self.assertGreater(min(depth),0)
if __name__=='__main__':unittest.main()
